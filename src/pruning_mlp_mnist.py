"""
Target-Class-Oriented Global Neuron Pruning for a Keras Dense MNIST Model
-------------------------------------------------------------------------

This script generalizes the original pruning code while staying faithful to its
core logic. It:
  • Loads a Keras .h5 model file and reads layer kernels via h5py (as in the original).
  • Keeps only the Target Class(es) of Interest (TCoI) in the output layer.
  • Computes per-neuron Importance Scores (IS) by back-propagating importance
    from the output to earlier hidden layers using column-normalized |weights|.
  • Iteratively prunes the globally least-important neuron across hidden layers.
  • Applies the pruning decisions using Kerassurgeon and saves pruned models.
  • The model is a feed-forward Dense network with an output layer (e.g., 10 units)
    preceded by three Dense hidden layers (e.g., 100, 400, 400 units).
  • Layer indices used by Kerassurgeon are provided.
  • The h5 structure contains "model_weights/<layer>/<layer>/kernel:0".
    (This is typical for older Keras .h5 files.)
"""

# Import core Keras APIs for loading/compiling the model
from keras.models import Sequential,load_model
# Import Adam optimizer (original API; uses 'lr' argument below)
from keras.optimizers import Adam
# Import Kerassurgeon to physically delete Dense units ("channels") from layers
from kerassurgeon import Surgeon
# h5py is used to open the .h5 file and read raw weight tensors
import h5py
# NumPy for array manipulation and vectorized math
import numpy as np
# (Kept for fidelity) L2 norm import; not used in this script but present in original
from numpy.linalg import norm


def NewPruning(TCoI):
    # Function performing TCoI-driven global pruning for a single target class index

    # Base model filename (without extension); must exist under the given directory
    weights_name = 'MLP_Model'

    # Open the saved Keras .h5 model file in read-only mode to access raw weights
    f = h5py.File(weights_name + '.h5', 'r')

    # List to hold absolute value weight matrices per connection (output-connection first)
    ArrLayersWeights=[]
    # Iterate over model_weight groups in reverse order, skipping last two groups (fidelity to original)
    for l in reversed(list(f['model_weights'])[:-2]):  
        # Access inner path model_weights/<layer>/<layer>/kernel:0 to get the kernel tensor
        data = f['model_weights'][l][l]['kernel:0']
        # Convert HDF5 dataset to NumPy array
        w = np.array(data)
        # Use absolute values of weights to ignore sign and measure magnitude-based contribution
        df1 = np.abs(w)  
        ArrLayersWeights.append(df1) 

    # Create one IS (importance score) vector per connection; dtype=object to store variable-length arrays
    ArrLayersIS = np.zeros(len(ArrLayersWeights),dtype=object)
    for kkk in range(len(ArrLayersIS)):
        # IS vector length equals number of output units (columns) of this connection
        ArrTemporary = np.zeros((1, len(ArrLayersWeights[kkk][1,:])))
        ArrLayersIS[kkk] = ArrTemporary[0] 

    # Create one prune-mask vector per connection (1=active, 0=pruned) initialized to ones
    ArrLayersPrunedNeuron = np.zeros(len(ArrLayersWeights),dtype=object)
    for lll in range(len(ArrLayersIS)):
        # Mask length equals number of output units (columns) of this connection
        ArrTemporary2= np.ones((1, len(ArrLayersWeights[lll][1,:])))
        ArrLayersPrunedNeuron[lll]= ArrTemporary2[0] 

    # Build a 10-way output mask (MNIST-style) and keep only the TCoI index
    ArrOutputLayersTcoi = np.array([0,0,0,0,0,0,0,0,0,0])
    # Store TCoI(s) in a list; here only a single class is added
    output_layer_TCoI = []
    output_layer_TCoI.append(TCoI)

    # Set the chosen class index to 1 to mark it as kept
    ArrOutputLayersTcoi[TCoI]=1

    # Apply the output mask to the prune mask of the output-connection (index 0 in our arrays)
    ArrLayersPrunedNeuron[0] = ArrOutputLayersTcoi  



    # Helper to avoid division-by-zero; kept for fidelity (not used directly below)
    def zero_divide(x, y):
        return 0 if y == 0 else x / y

    # Initialize output-layer IS equal to the TCoI mask (kept classes have IS=1)
    ArrOutputLayerIS = ArrOutputLayersTcoi  
    ArrLayersIS[0] = ArrOutputLayersTcoi  

    # Temporary collectors to track smallest positive IS per layer during selection
    ListTemporaryMinIndex=[]
    ListTemporaryMin = []

    # Running counter of how many neurons have been pruned so far (for naming outputs)
    name_pruned_neuron_number = 0
    # Outer loop: prune in rounds; each entry indicates how many neurons to prune in that round
    # Outer loop: prune neurons in fixed-size groups (50 each) to empirically find
    # the point up to which pruning maintains performance before degradation starts.
    for NeuronNumberToPrune in [50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]:
        # Inner loop: prune exactly NeuronNumberToPrune neurons, one-by-one (global smallest IS)
        for i in range(NeuronNumberToPrune): 
            # Apply current prune masks to weight matrices (broadcast multiply preserves shape)
            ArrLayersWeights = np.multiply(ArrLayersWeights, ArrLayersPrunedNeuron)

            # Additionally mask columns by downstream layer masks to cut deactivated targets
            ArrLayersWeights[0]= ArrLayersPrunedNeuron[1].reshape((len(ArrLayersPrunedNeuron[1]), 1))*ArrLayersWeights[0]
            ArrLayersWeights[1]= ArrLayersPrunedNeuron[2].reshape((len(ArrLayersPrunedNeuron[2]), 1))*ArrLayersWeights[1]
            ArrLayersWeights[2]= ArrLayersPrunedNeuron[3].reshape((len(ArrLayersPrunedNeuron[3]), 1))*ArrLayersWeights[2]

            # For each connection except the last, normalize columns and back-propagate IS
            for kk in range(len(ArrLayersWeights)-1):#[0,1,3]

                # Column sums for normalization; avoids scale bias in IS propagation
                axis0Sum = ArrLayersWeights[kk].sum(axis=0)  
                # In-place safe division: only where column sum is non-zero
                np.divide(ArrLayersWeights[kk], axis0Sum,
                          out=ArrLayersWeights[kk],
                          where=axis0Sum != 0)

                # Back-propagate importance: IS_prev = sum_j( W_norm[i,j] * IS_curr[j] )
                ArrLayersIS[kk+1]=(ArrLayersWeights[kk]*ArrLayersIS[kk]).sum(axis=1)


            # From hidden-layer IS vectors, pick the smallest positive IS globally
            for rrr in ArrLayersIS[1:]:
                # Skip zeros (already pruned or inactive); take the minimum positive value
                min_val = min(i for i in rrr if i > 0)
                ListTemporaryMin.append(min_val)
                # Record index of this minimum in the layer
                index_min = np.where(rrr == min_val )
                ListTemporaryMinIndex.append(index_min)

            # Convert to arrays for argmin across layers
            ListTemporaryMin      = np.array(ListTemporaryMin)
            ListTemporaryMinIndex = np.array(ListTemporaryMinIndex)
            # Find which layer has the global minimum IS (across stored minima)
            MinOfTheMinimumsIndex = np.argmin(ListTemporaryMin)
            # Set that neuron's mask to 0 (prune) in the corresponding hidden layer
            ArrLayersPrunedNeuron[MinOfTheMinimumsIndex+1][ListTemporaryMinIndex[MinOfTheMinimumsIndex]] = 0
            # (Optional) The pruned layer index (1-based in our hidden-layer indexing)
            LayerNumber= MinOfTheMinimumsIndex +1

            # Reset collectors for next iteration
            ListTemporaryMinIndex = []
            ListTemporaryMin = []



        # After finishing this round, gather pruned indices per hidden layer for surgeon
        All_Pruned_Index_Value=[]
        for zzz in ArrLayersPrunedNeuron[1:]:
                # Walk through the mask and collect indices where value==0 (pruned)
                IndexNumber=0
                TempList=[]
                for z in zzz:
                    if z==0:
                        TempList.append(IndexNumber)
                    IndexNumber = IndexNumber + 1
                # Log current layer's pruned indices
                print('Temp List = ',TempList)
                All_Pruned_Index_Value.append(TempList)
                # Reset temporary structures
                TempList = []
                IndexNumber = 0
        # Reload the original (unmodified) model to apply surgery on a fresh graph
        model=load_model(weights_name + '.h5')
        model.summary()

        # Resolve layer objects by index (architecture-specific; matches original model)
        layer_10_output=model.layers[5]
        layer_100=model.layers[4]
        layer_400_2=model.layers[2]
        layer_400_1=model.layers[0]
        # Create a Surgeon to queue and execute delete_channels operations
        surgeon=Surgeon(model)
        # In the output layer, delete all classes except TCoI (keep only indices in output_layer_TCoI)
        surgeon.add_job('delete_channels',layer_10_output,channels=[i for i in range(10) if i not in output_layer_TCoI]) 
        # Delete pruned neuron indices in each hidden layer (from earlier masks)
        surgeon.add_job('delete_channels',layer_100,channels=All_Pruned_Index_Value[0])
        surgeon.add_job('delete_channels',layer_400_2,channels= All_Pruned_Index_Value[1])
        surgeon.add_job('delete_channels',layer_400_1,channels=All_Pruned_Index_Value[2])
        # Execute the surgery to obtain the physically pruned model
        model_new=surgeon.operate()
        model_new.summary()

        # Compile the pruned model (same optimizer/settings pattern as original)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Update the running count for filename generation
        name_pruned_neuron_number = name_pruned_neuron_number +  NeuronNumberToPrune

        # Build a descriptive filename including TCoI and total pruned count so far
        weights_name_pruned = ('%d'%(TCoI)) + ('_%d' %(name_pruned_neuron_number))
        # Save the pruned model under the specified output directory
        model_new.save( weights_name_pruned + '.h5')

# Define which target classes to keep; code will run pruning once per TCoI
TCoIs = [8] #[0,1,2,3,4,5,6,7,8,9]

# Driver loop: invoke pruning for each selected TCoI
for j in TCoIs:
    NewPruning(j)

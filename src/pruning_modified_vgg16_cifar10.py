

# Load a VGG16 application model class (not directly used below; safe to remove import)
from tensorflow.keras.applications.vgg16 import VGG16
# Helpers to sort tuples/lists by fields
from operator import itemgetter, attrgetter
# NumPy core
import numpy as np
# Kerassurgeon: we'll use this to physically delete channels/filters/neurons
from kerassurgeon import Surgeon
# Optimizer import (used later to compile the pruned model)
from tensorflow.keras.optimizers import Adam
# Generic warnings control (we silence warnings for cleaner logs)
import warnings
# For printing full arrays without truncation
import sys
# Timing (some commented timing code lower in file references this)
import time

# Print full NumPy arrays; avoid "..." truncation in logs
np.set_printoptions(threshold=sys.maxsize)
# Silence warnings (optional)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset note (for context only):
# CIFAR-10 classes = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
# 50,000 train + 10,000 test, images are 32x32x3, 10 classes.
# ---------------------------------------------------------------------------

# We will load a saved Keras model and inspect/prune it
from numpy import loadtxt
# NOTE: This 'load_model' comes from *keras.models* (not tf.keras). Your model
# was saved in a way that this loader can read. Keep consistent with how it was saved.
from keras.models import load_model

# ---------------------------------------------------------------------------
# Load your pretrained Modified VGG16 model (CIFAR-10) from disk
# ---------------------------------------------------------------------------
model_ModifiedVGG16 = load_model('Modified_VGG16_CIFAR10.h5')
# Print model architecture summary (helps verify layer indices used later)
model_ModifiedVGG16.summary()

# ---------------------------------------------------------------------------
# Quick scan: print the name and weight shape of each convolutional layer
# This confirms how many filters and what shapes we will be dealing with.
# ---------------------------------------------------------------------------
for layer in model_ModifiedVGG16.layers:
    # check for convolutional layer by name
    if 'conv' not in layer.name:
        continue
    # get filter weights (kernel) and biases from this conv layer
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)


def get_filter_weights(model, layer=None):
    """Return kernel weights for one or for all convolutional layers.
    If 'layer' is an integer, return the kernel of that specific layer index.
    Otherwise, return a list of kernels for all layers whose name contains 'conv'.
    """
    if layer or layer == 0:  # single layer path (kept original logic)
        weight_array = model.layers[layer].get_weights()[0]
    else:
        weights = [
            model.layers[layer_ix].get_weights()[0]
            for layer_ix in range(len(model.layers))
            if 'conv' in model.layers[layer_ix].name
        ]
        weight_array = [np.array(i) for i in weights]
    return weight_array




# ---------------------------------------------------------------------------
# Fully Connected (Dense) layers: grab absolute weights of the last FC blocks
# Here, layers [22, 21, 20] are assumed to be [output, FC2, FC1] respectively
# (based on your specific model). We take abs(weights) 
# ---------------------------------------------------------------------------
ListAbsolutedFCL_new = []
for layer_ix in [22, 21, 20]:
    weights = model_ModifiedVGG16.layers[layer_ix].get_weights()
    raw_data_of_a_layer = weights[0]  # take kernel only, drop bias
    # Absolute weights for magnitude-based contributions
    absoluted_weight = np.abs(raw_data_of_a_layer, dtype=np.float32)


    print('sum of FCL layer = ', np.sum(absoluted_weight))
    # Collect abs-weight matrix for downstream IS propagation
    ListAbsolutedFCL_new.append(absoluted_weight)

# ---------------------------------------------------------------------------
# Build a single IS (importance score) structure across FC + Conv sections
# There are 3 FC layers and 13 Conv layers = 16 entries in ArrLayersIS.
# Each entry is a vector of size equal to the number of outputs of that layer.
# ---------------------------------------------------------------------------
ArrLayersIS = np.zeros(16, dtype=object)  # 3 FC + 13 Conv = 16 slots
# Initialize IS vectors for FC layers according to their output sizes
for kkk in range(3):
    ArrTemporaryFCL = np.zeros((1, len(ListAbsolutedFCL_new[kkk][1, :])))
    ArrLayersIS[kkk] = ArrTemporaryFCL[0]  # keep as 1D row

# ---------------------------------------------------------------------------
# Get convolutional kernels from the model (all conv layers)
# ---------------------------------------------------------------------------
weight_array = get_filter_weights(model_ModifiedVGG16, layer=None)

# Initialize IS vectors for Conv layers according to their #filters (output channels)
counter1 = 3
for i in reversed(weight_array):
    ArrTemporaryConv = np.zeros((1, i.shape[3]), dtype=object)
    ArrLayersIS[counter1] = ArrTemporaryConv[0]
    counter1 = counter1 + 1

# ---------------------------------------------------------------------------
# Precompute per-layer channel sums for Conv layers:
# For each conv layer, we sum absolute kernel weights per (input channel, output filter).
# This is used later in IS propagation to distribute importance through channels.
# ---------------------------------------------------------------------------
ArrWeighsArraySumOfChannels = np.empty(len(weight_array), dtype=object)
counter_for_channels_sum = 0
for i in reversed(weight_array):
    i = np.abs(i, dtype=np.float32)
    

    # Build [in_channels x out_filters] matrix of summed magnitudes over spatial dims
    ArrSumofFiltersChannels = np.zeros((i.shape[2], i.shape[3]), dtype=object)
    for s in range(i.shape[2]):      # input channel index
        for k in range(i.shape[3]):  # output filter index
            Channel = 0
            for l in range(i.shape[1]):      # kernel height
                for m in range(i.shape[0]):  # kernel width
                    Channel = Channel + (i[m][l][s][k])
            ArrSumofFiltersChannels[s, k] = Channel
            Channel = 0
    ArrWeighsArraySumOfChannels[counter_for_channels_sum] = ArrSumofFiltersChannels
    counter_for_channels_sum = counter_for_channels_sum + 1

# ---------------------------------------------------------------------------
# Build TCoI (Target Class of Interest) mask for the output layer.
# CIFAR-10 example mapping: car=1, dog=5, frog=6, truck=9 (your note).
# ---------------------------------------------------------------------------
ArrOutputLayersTcoi = np.zeros(10, dtype=np.float32)
TCoI = 1  # choose which class to keep at output (e.g., "automobile"=1)
ArrOutputLayersTcoi[TCoI] = 1

# ---------------------------------------------------------------------------
# Initialize prune masks (1=keep, 0=prune) per layer:
# 16 entries matching ArrLayersIS (3 FC + 13 Conv).
# Output layer mask is set to TCoI; IS[0] also seeded with TCoI mask.
# ---------------------------------------------------------------------------
ArrLayersPrunedNeuron = np.zeros(16, dtype=object)
for i in range(len(ArrLayersPrunedNeuron)):
    ArrLayersPrunedNeuron[i] = np.ones_like(ArrLayersIS[i], dtype=np.float32)

ArrLayersPrunedNeuron[0] = ArrOutputLayersTcoi.copy()
ArrLayersIS[0] = ArrOutputLayersTcoi.copy()

def foo(x, y):
    # Division helper (kept for fidelity with the original; not used below)
    return 0 if y == 0 else x / y

# Each cycle will prune 'how_many_neuron_per_cycle' neurons (see call below)
name_pruned_neuron_number = 1  # per-cycle prune amount (passed later)
# how_many_times_cycle = 3500  # (unused example; kept as comment)

# ---------------------------------------------------------------------------
# Main pruning routine:
#  - Propagate IS through FC and Conv sections with current prune masks.
#  - Mix IS with entropy (layer-wise factor).
#  - Select smallest-IS neurons/filters globally and mark them pruned.
#  - After finishing 'cycle' iterations, apply surgery and save the pruned model.
# ---------------------------------------------------------------------------
def Pruning(cycle, how_many_neuron_per_cycle, tcoi, PrunedNeuron):

    for i in range(cycle):  # perform the pruning-selection loop 'cycle' times
        # ----- FC SECTION: apply current prune masks on FC weights (left/right sides) -----
        # Copy the absolute FC weights fresh each iteration
        ListAbsolutedFCL = ListAbsolutedFCL_new.copy()
        # Left-side masking (zero columns for pruned targets)
        for ihn in range(3):
            ListAbsolutedFCL[ihn] = ListAbsolutedFCL[ihn] * PrunedNeuron[ihn]

        # Right-side masking (zero rows for pruned sources feeding into next FC layer)
        for ihnm in range(2):
            ListAbsolutedFCL[ihnm] = PrunedNeuron[ihnm + 1].reshape(
                (len(PrunedNeuron[1]), 1)
            ) * ListAbsolutedFCL[ihnm]

        # Propagate pruning from last conv layer to the first FC after flatten:
        # If a conv filter is pruned, its corresponding flattened entries are zeroed.
        counter_for_pruned = 0
        featuremapsize = 1*1  # with your Modified VGG, flatten featuremap size is 1x1 here
        for pruned_index in PrunedNeuron[3]:
            if pruned_index == 0:
                ListAbsolutedFCL[2][
                    (counter_for_pruned * featuremapsize):(counter_for_pruned * featuremapsize + featuremapsize)
                ] = 0
            counter_for_pruned = counter_for_pruned + 1

        # Column-normalize FC weights, multiply by current IS, and sum across columns
        # to get IS for the preceding layer (classic importance back-propagation step).
        for o in range(len(ListAbsolutedFCL)):  # [0,1,2]
            axis0Sum = ListAbsolutedFCL[o].sum(axis=0).copy()
            np.divide(ListAbsolutedFCL[o], axis0Sum, out=ListAbsolutedFCL[o], where=axis0Sum != 0)
            np.multiply(
                ListAbsolutedFCL[o], ArrLayersIS[o], out=ListAbsolutedFCL[o]
            )
            ArrLayersIS[o + 1] = ListAbsolutedFCL[o].sum(axis=1).copy()

        # ----- FLATTEN TO CONV BRIDGE: aggregate IS over the featuremap tiles -----
        ListFirstConvLayerFiltersIS = []
        flatten_layer_size = len(ArrLayersIS[3])
        plus_forty_nine = 0  # (for a 7x7 FM this would be 49; here featuremapsize=1)
        for imk in range(int(flatten_layer_size / featuremapsize)):
            ListFirstConvLayerFiltersIS.append(
                sum(ArrLayersIS[3][plus_forty_nine:(plus_forty_nine + featuremapsize)])
            )
            plus_forty_nine = plus_forty_nine + featuremapsize
        ArrLayersIS[3] = np.array(ListFirstConvLayerFiltersIS, dtype=np.float32)

        # ----- CONV SECTION: propagate IS through conv stacks using pre-summed channels -----
        n = 4  # IS index for the first conv layer in our unified structure
        for i in ArrWeighsArraySumOfChannels:
            ArrSumofFiltersChannels = np.float32(i)

            # Mask with previously pruned filters (left side)
            ArrSumofFiltersChannels = ArrSumofFiltersChannels * (
                PrunedNeuron[n - 1].reshape((1, len(PrunedNeuron[n - 1])))
            )
            # Mask with pruned filters on the right (next layer), if not the last conv stage
            if n != 16:
                ArrSumofFiltersChannels = ArrSumofFiltersChannels * (
                    PrunedNeuron[n].reshape((len(PrunedNeuron[n]), 1))
                )

            # Normalize per-filter by channel sums to dampen scale effects
            ArrSumofFiltersweight = ArrSumofFiltersChannels.sum(axis=0).copy()
            reshaped = ArrSumofFiltersweight.reshape((1, len(ArrSumofFiltersweight)))
            np.divide(
                ArrSumofFiltersChannels, reshaped, out=ArrSumofFiltersChannels, where=reshaped != 0
            )
            # Multiply by upstream IS and sum to get this layer's IS
            ArrSumofFiltersChannels = ArrSumofFiltersChannels * (
                ArrLayersIS[n - 1].reshape((1, len(ArrLayersIS[n - 1])))
            ).copy()
            if n != 16:
                ArrLayersIS[n] = ArrSumofFiltersChannels.sum(axis=1)
            n = n + 1



        # ----- GLOBAL SELECTION: sort all (layer, index, IS) tuples by IS ascending -----
        ListAllForMinVal = []
        ListForMinValue = []
        index1 = 1
        bitis_index = 12  # (unused variable from original; kept)
        for r in ArrLayersIS[index1:]:
            index2 = 0
            for rr in r:
                ListForMinValue.append([index1, index2, rr])
                index2 = index2 + 1
            index1 = index1 + 1
            index2 = 0

        # Zero-out non-output IS temporarily (to avoid re-selecting already pruned entries)
        index1 = 1
        for i in templist:
            ArrLayersIS[i] *= 0
        ArrLayersIS[0] = ArrOutputLayersTcoi.copy()

        # Sort candidates by IS (ascending) and drop zeros
        ListSorted = sorted(ListForMinValue, key=itemgetter(2))
        ListSortedWithoutZeros = []
        for ink in ListSorted:
            if ink[2] != 0:
                ListSortedWithoutZeros.append(ink)

        # Prune the lowest-IS entries across layers (batch size = how_many_neuron_per_cycle)
        for t in ListSortedWithoutZeros[:how_many_neuron_per_cycle]:
            print(t)
            PrunedNeuron[t[0]][t[1]] = 0

    # -----------------------------------------------------------------------
    # After finishing 'cycle' iterations: collect pruned indices per layer,
    # perform surgery with Kerassurgeon, compile, and save the pruned model.
    # -----------------------------------------------------------------------
    sum_of_pruned_neuron = 0
    All_Pruned_Index_Value = []
    for zzz in PrunedNeuron[1:]:
        IndexNumber = 0
        TempList = []
        for z in zzz:
            if z == 0:
                TempList.append(IndexNumber)
            IndexNumber = IndexNumber + 1
        print('Temp List = ', TempList)
        sum_of_pruned_neuron = sum_of_pruned_neuron + len(TempList)
        All_Pruned_Index_Value.append(TempList)
    print('sum_of_pruned_neuron', sum_of_pruned_neuron)

    # Map specific layers by index (architecture-specific to your Modified VGG16)
    model = model_ModifiedVGG16
    layer_10_output = model.layers[22]
    fc2_4096 = model.layers[21]
    fc1_4096 = model.layers[20]
    block5_conv3 = model.layers[17]
    block5_conv2 = model.layers[16]
    block5_conv1 = model.layers[15]
    block4_conv3 = model.layers[13]
    block4_conv2 = model.layers[12]
    block4_conv1 = model.layers[11]
    block3_conv3 = model.layers[9]
    block3_conv2 = model.layers[8]
    block3_conv1 = model.layers[7]
    block2_conv2 = model.layers[5]
    block2_conv1 = model.layers[4]
    block1_conv2 = model.layers[2]
    block1_conv1 = model.layers[1]

    # Create surgeon and queue delete jobs:
    # - At output: keep only 'tcoi' class; delete all other class channels.
    surgeon = Surgeon(model)
    surgeon.add_job(
        'delete_channels',
        layer_10_output,
        channels=[ipk for ipk in range(10) if ipk != tcoi]
    )

    # - FC layers: delete pruned neurons (indices collected above)
    surgeon.add_job('delete_channels', fc2_4096, channels=All_Pruned_Index_Value[0])
    surgeon.add_job('delete_channels', fc1_4096, channels=All_Pruned_Index_Value[1])

    # - CONV layers: delete pruned filters per block (deep to shallow)
    surgeon.add_job('delete_channels', block5_conv3, channels=All_Pruned_Index_Value[2])
    surgeon.add_job('delete_channels', block5_conv2, channels=All_Pruned_Index_Value[3])
    surgeon.add_job('delete_channels', block5_conv1, channels=All_Pruned_Index_Value[4])

    surgeon.add_job('delete_channels', block4_conv3, channels=All_Pruned_Index_Value[5])
    surgeon.add_job('delete_channels', block4_conv2, channels=All_Pruned_Index_Value[6])
    surgeon.add_job('delete_channels', block4_conv1, channels=All_Pruned_Index_Value[7])

    surgeon.add_job('delete_channels', block3_conv3, channels=All_Pruned_Index_Value[8])
    surgeon.add_job('delete_channels', block3_conv2, channels=All_Pruned_Index_Value[9])
    surgeon.add_job('delete_channels', block3_conv1, channels=All_Pruned_Index_Value[10])

    surgeon.add_job('delete_channels', block2_conv2, channels=All_Pruned_Index_Value[11])
    surgeon.add_job('delete_channels', block2_conv1, channels=All_Pruned_Index_Value[12])

    surgeon.add_job('delete_channels', block1_conv2, channels=All_Pruned_Index_Value[13])
    surgeon.add_job('delete_channels', block1_conv1, channels=All_Pruned_Index_Value[14])

    # Apply all deletions and get the pruned model
    model_new = surgeon.operate()
    model_new.summary()

    # Compile the pruned model (same hyperparams pattern you used elsewhere)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Save with a filename reflecting the total number of pruned units
    weights_name_pruned = '%d' % (sum_of_pruned_neuron)
    model_new.save(weights_name_pruned + '.h5')
    return PrunedNeuron

# ---------------------------------------------------------------------------
# PRUNING SCHEDULE:
# We call Pruning() multiple times with different 'cycle' values.
# The idea is staged/grouped pruning to observe how far we can prune
# while maintaining performance (accuracy-preserving pruning budget).
#   - First pass: cycle=2200 (aggressive)
#   - Next passes: cycle=50, 50, 50 (fine-grained top-ups)
# Each 'cycle' will prune 'name_pruned_neuron_number' entries per iteration.
# ---------------------------------------------------------------------------
for i in [2200,50,50,50]:
    ArrLayersPrunedNeuron = Pruning(i, name_pruned_neuron_number, TCoI, ArrLayersPrunedNeuron)
    print('ArrLayersPrunedNeuron',ArrLayersPrunedNeuron)

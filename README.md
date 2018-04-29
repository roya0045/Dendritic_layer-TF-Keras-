# Dendritic_layer-TF-Keras-
Dendritic neural network layer inspired by 
https://www.researchgate.net/publication/323998838_Adaptive_nodes_enrich_nonlinear_cooperative_learning_beyond_traditional_adaptation_by_links

This is a prototype layer.
If you decide to try it out please inform me of the results as I do not have much time for testing currently.

Some tweaks must still be done, this is a work in progress

The TF implementation has 4 version.

Version 1 is currently broken due to how unsorted_segment_sum currently works, if the alg is updated to preseve the first dimentions if would work, hence why it remains as of now.

	??%䃞????%䃞??!??%䃞??	?z??@?z??@!?z??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??%䃞??ŏ1w-??Ax??#????Y??A?f??*	     ^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?g??s???!?旁??A@)?0?*???15?5?@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?{??Pk??!m)im)iE@)??ܵ?|??1sԸrԸ:@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?j+??ݓ?!h~h~0@)?j+??ݓ?1h~h~0@:Preprocessing2U
Iterator::Model::ParallelMapV2_?Qڋ?!ז?֖?&@)_?Qڋ?1ז?֖?&@:Preprocessing2F
Iterator::Model
ףp=
??!??>?2@)??y?):??1N'?M'?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipJ+???!??T??TT@)?~j?t?h?1T??S??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!?,6?,6??)/n??b?1?,6?,6??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	?c???!?b??b?E@)????MbP?1W?W???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t18.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?z??@IRؗN??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ŏ1w-??ŏ1w-??!ŏ1w-??      ??!       "      ??!       *      ??!       2	x??#????x??#????!x??#????:      ??!       B      ??!       J	??A?f????A?f??!??A?f??R      ??!       Z	??A?f????A?f??!??A?f??b      ??!       JCPU_ONLYY?z??@b qRؗN??W@
	??ǘ??????ǘ????!??ǘ????	?΍j??<@?΍j??<@!?΍j??<@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ǘ????5?8EGr??A?V-??Y??????*23333h@)       =2F
Iterator::Model?!??u???!?????+U@)Έ?????1?ʂ\?PS@:Preprocessing2U
Iterator::Model::ParallelMapV2???QI??!?1W??@)???QI??1?1W??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Q???!???'@)?{??Pk??1?{%??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?j+??݃?!?I?7%@)S?!?uq{?1v??$g?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph??|?5??!j?P?(?.@)??H?}m?1*J?#???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?h?!yh=????)?~j?t?h?1yh=????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!??D*r??)?J?4a?1??D*r??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<?R??!??'9?@)a2U0*?S?1ǆ?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 28.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s7.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?΍j??<@I`?\???Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5?8EGr??5?8EGr??!5?8EGr??      ??!       "      ??!       *      ??!       2	?V-???V-??!?V-??:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JCPU_ONLYY?΍j??<@b q`?\???Q@
package org.apache.mahout.classifier.sgd.wikipedia;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.MultiLabelVectorWritable;

public class WikipediaIdentityReducer extends MapReduceBase
    implements
    Reducer<LongWritable,MultiLabelVectorWritable,LongWritable,MultiLabelVectorWritable> {

  @Override
  public void reduce(LongWritable key,
      Iterator<MultiLabelVectorWritable> values,
      OutputCollector<LongWritable,MultiLabelVectorWritable> collector,
      Reporter reporter) throws IOException {
    while (values.hasNext()) {
      collector.collect(key, values.next());
    }
  }
}

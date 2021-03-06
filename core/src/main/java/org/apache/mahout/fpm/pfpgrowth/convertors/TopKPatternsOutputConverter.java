/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.fpm.pfpgrowth.convertors;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FrequentPatternMaxHeap;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.Pattern;

/**
 * An output converter which converts the output patterns and collectes them in a
 * {@link FrequentPatternMaxHeap}
 * 
 * @param <A>
 */
public final class TopKPatternsOutputConverter<A extends Comparable<? super A>> implements
    OutputCollector<Integer,FrequentPatternMaxHeap> {
  
  private OutputCollector<A,List<Pair<List<A>,Long>>> collector;
  
  private final Map<Integer,A> reverseMapping;
  
  public TopKPatternsOutputConverter(OutputCollector<A,List<Pair<List<A>,Long>>> collector,
                                     Map<Integer,A> reverseMapping) {
    this.collector = collector;
    this.reverseMapping = reverseMapping;
  }
  
  @Override
  public void collect(Integer key, FrequentPatternMaxHeap value) throws IOException {
    List<Pair<List<A>,Long>> perAttributePatterns = new ArrayList<Pair<List<A>,Long>>();
    PriorityQueue<Pattern> t = value.getHeap();
    while (!t.isEmpty()) {
      Pattern itemSet = t.poll();
      List<A> frequentPattern = new ArrayList<A>();
      for (int j = 0; j < itemSet.length(); j++) {
        frequentPattern.add(reverseMapping.get(itemSet.getPattern()[j]));
      }
      Collections.sort(frequentPattern);
      
      Pair<List<A>,Long> returnItemSet = new Pair<List<A>,Long>(frequentPattern, itemSet.support());
      perAttributePatterns.add(returnItemSet);
    }
    Collections.reverse(perAttributePatterns);
    
    collector.collect(reverseMapping.get(key), perAttributePatterns);
  }
}

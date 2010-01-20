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

package org.apache.mahout.classifier.sgd.wikipedia;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.wikipedia.analysis.WikipediaTokenizer;
import org.apache.mahout.classifier.sgd.BinaryRandomizer;
import org.apache.mahout.classifier.sgd.TermRandomizer;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.MultiLabelVectorWritable;

/**
 * Vectorize Wikipedia articles and use the categories tags as labels. The key
 * of the emitted intermediate data is voluntarily randomized so to meet the
 * I.I.D. assumption of the stochastic learner living in the reducer.
 *
 */
public class WikipediaRandomHasherMapper extends MapReduceBase implements
    Mapper<LongWritable,Text,LongWritable,MultiLabelVectorWritable> {

  private static final Pattern OPEN_TEXT_TAG_PATTERN = Pattern
      .compile("<text xml:space=\"preserve\">");
  private static final Pattern CLOSE_TEXT_TAG_PATTERN = Pattern
      .compile("</text>");

  private List<String> inputCategories;
  private double maxUnlabeledInstanceRate;
  private long unlabeledOuputCount = 0; // use shared counters instead?
  private long totalOutputCount = 0;
  private int seed = 42;
  private Random rng;
  private TermRandomizer randomizer;
  private boolean allPairs;
  private int window;
  private final LongWritable shufflingKey = new LongWritable();
  private final MultiLabelVectorWritable labeledVectorValue = new MultiLabelVectorWritable();

  @Override
  public void configure(JobConf job) {
    // seed the RNG used to shuffle the instances (wikipedia articles come in
    // Alphabetical order and that bias could harm the convergence of online
    // learner that assume I.I.D. samples).
    seed = job.getInt("wikipedia.random.seed", seed);
    rng = RandomUtils.getRandom(seed);

    // load the list of category labels to look for
    String categoriesStr = job.get("wikipedia.categories", "");
    inputCategories = Arrays.asList(categoriesStr.split(","));

    // reasonable default to avoid generating to many unlabeled instances
    maxUnlabeledInstanceRate = 1.0 / inputCategories.size();

    // load the randomizer that is used to hash the term of the document
    int probes = job.getInt("randomizer.probes", 2);
    int numFeatures = job.getInt("randomizer.numFeatures", 80000);
    randomizer = new BinaryRandomizer(probes, numFeatures);
    allPairs = job.getBoolean("randomizer.allPairs", false);
    window = job.getInt("randomizer.window", 2);

  }

  @Override
  public void map(LongWritable key, Text value,
      OutputCollector<LongWritable,MultiLabelVectorWritable> collector,
      Reporter reporter) throws IOException {
    shufflingKey.set(rng.nextLong());

    // extract the raw markup from the XML dump slice
    String document = StringEscapeUtils.unescapeHtml(CLOSE_TEXT_TAG_PATTERN
        .matcher(
            OPEN_TEXT_TAG_PATTERN.matcher(value.toString()).replaceFirst(""))
        .replaceAll(""));

    // collect the categories as indexes
    int[] categories = findMatchingCategories(document);
    labeledVectorValue.setLabels(categories);

    // ensure we are not
    if (categories.length == 0) {
      if (totalOutputCount != 0
          && ((double) unlabeledOuputCount) / totalOutputCount > maxUnlabeledInstanceRate) {
        return;
      }
      unlabeledOuputCount++;
    }
    totalOutputCount++;

    // strip the wikimarkup and hash the terms using the randomizer
    TokenStream stream = new WikipediaTokenizer(new StringReader(document));
    List<String> allTerms = new ArrayList<String>();
    TermAttribute termAtt = (TermAttribute) stream
        .addAttribute(TermAttribute.class);
    while (stream.incrementToken()) {
      allTerms.add(termAtt.term());
    }
    // TODO: refactor randomizedInstance to take a token stream as input and
    // avoid all those wasted string allocations (or prove they are harmless
    // using the profiler).
    labeledVectorValue.set(randomizer.randomizedInstance(allTerms, window,
        allPairs));
    collector.collect(shufflingKey, labeledVectorValue);
  }

  private int[] findMatchingCategories(String document) {
    List<Integer> matchingCategories = new ArrayList<Integer>();
    int startIndex = 0;
    int categorystart;
    while ((categorystart = document.indexOf("[[Category:", startIndex)) != -1) {
      categorystart += 11;
      int endIndex = document.indexOf("]]", categorystart);
      if (endIndex >= document.length() || endIndex < 0) {
        break;
      }
      String category = document.substring(categorystart, endIndex)
          .toLowerCase().trim();
      if (inputCategories.contains(category)) {
        matchingCategories.add(inputCategories.indexOf(category));
      }
      startIndex = endIndex;
    }
    return ArrayUtils.toPrimitive(matchingCategories
        .toArray(new Integer[matchingCategories.size()]));
  }

}

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

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.bayes.XmlInputFormat;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.math.MultiLabelVectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extract a vectorized dataset from a chunked Wikipedia XML dump. The generated
 * dataset is suitable for the training of a Logistic Regression model using
 * Stochastic Gradient Descent so as to perform document classification.
 *
 * For each document the list of labels are extracted the integer indexes of the
 * document categories if the belong to the list of categories given as input.
 *
 * The features are extracted from the tokenized content of each wiki articled
 * deterministically mapped to a fixed dimensional array of occurrence and
 * coocurrence of terms counts through a random function (a.k.a TermRandomizer).
 */
public final class WikipediaHashedDatasetCreatorDriver extends Configured
    implements Tool {
  private static final Logger log = LoggerFactory
      .getLogger(WikipediaHashedDatasetCreatorDriver.class);

  private WikipediaHashedDatasetCreatorDriver() {}

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(),
        new WikipediaHashedDatasetCreatorDriver(), args);
    System.exit(res);
  }

  public int run(String[] args) {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option dirInputPathOpt = obuilder.withLongName("input").withRequired(true)
        .withArgument(
            abuilder.withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription("The input directory path").withShortName("i")
        .create();

    Option dirOutputPathOpt = obuilder.withLongName("output")
        .withRequired(true).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("The output directory Path").withShortName("o")
        .create();

    Option categoriesOpt = obuilder
        .withLongName("categories")
        .withRequired(true)
        .withArgument(
            abuilder.withName("categories").withMinimum(1).withMaximum(1)
                .create())
        .withDescription(
            "Location of the categories file.  One entry per line. "
                + "Will be used to make a string match in Wikipedia Category field")
        .withShortName("c").create();

    Option helpOpt = obuilder.withLongName("help").withDescription(
        "Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(categoriesOpt)
        .withOption(dirInputPathOpt).withOption(dirOutputPathOpt).withOption(
            helpOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    try {
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return 0;
      }

      String inputPath = (String) cmdLine.getValue(dirInputPathOpt);
      String outputPath = (String) cmdLine.getValue(dirOutputPathOpt);
      String catFile = (String) cmdLine.getValue(categoriesOpt);
      runJob(inputPath, outputPath, catFile);
      return 0;
    } catch (Exception e) {
      log.error(e.getMessage(), e);
      CommandLineUtil.printHelp(group);
      return 1;
    }
  }

  /**
   * Run the job
   *
   * @param input
   *          the input pathname String
   * @param output
   *          the output pathname String
   * @param catFile
   *          the file containing the Wikipedia categories, one entry per line
   */
  public static void runJob(String input, String output, String catFile)
      throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(WikipediaHashedDatasetCreatorDriver.class);
    if (log.isInfoEnabled()) {
      log.info("Input: " + input + " Out: " + output + " Categories: "
          + catFile);
    }
    conf.set("key.value.separator.in.input.line", " ");
    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(WikipediaRandomHasherMapper.class);
    conf.setReducerClass(IdentityReducer.class);
    conf.setInputFormat(XmlInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(MultiLabelVectorWritable.class);

    // we need to extract the title, not just the markup content
    conf.set(XmlInputFormat.START_TAG_KEY, "<page>");
    conf.set(XmlInputFormat.END_TAG_KEY, "</page>");

    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }

    Set<String> categories = new HashSet<String>();
    for (String line : new FileLineIterable(new File(catFile))) {
      categories.add(line.trim().toLowerCase());
    }
    conf.set("wikipedia.categories", StringUtils.join(categories, ','));
    client.setConf(conf);
    JobClient.runJob(conf);
  }

}

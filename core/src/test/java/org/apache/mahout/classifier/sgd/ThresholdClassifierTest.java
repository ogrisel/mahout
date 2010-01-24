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

package org.apache.mahout.classifier.sgd;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.mahout.common.RandomUtils;
import org.junit.BeforeClass;
import org.junit.Test;

public class ThresholdClassifierTest {

  public static final List<String> documents = Arrays
      .asList(
          // some German poetry by Annette von Droste-Hülshoff to make the
          // default precision poor enough to render the problem more
          // interesting
          "Ich steh' auf hohem Balkone am Turm",
          "Umstrichen vom schreienden Stare",
          "Und lass' gleich einer Mänade den Sturm",
          "Mir wühlen im flatternden Haare",
          "O wilder Geselle, o toller Fant",
          "Ich möchte dich kräftig umschlingen",
          "Und, Sehne an Sehne, zwei Schritte vom Rand",
          "Auf Tod und Leben dann ringen!",
          // Some sentences classified in 3 somewhat related topics + some
          // boring sentences
          "Online learning is incremental machine learning: instances are scanned"
              + " through and the algorithm updates the parameters on one instance at time.",
          "Clojure is an Open Source functional programming language with Lisp"
              + " syntax that compiles to Java bytecode and run on the JVM",
          "Batch learning as opposed to online learning is a class of machine learning"
              + " algorithms that updates the parameters after each complete scan of the dataset",
          "Apache Hadoop is an Open Source Java library that implements a distributed"
              + " filesystem and a MapReduce runtime for scalable data processing.",
          "This sentences is not very interesting",
          "Clustering is an unsupervised machine learning task: data instances do not carry labels",
          "Apache Mahout is an Open Source machine learning library written in Java that"
              + " leverages the Apache Hadoop project for scalable data processing.",
          "The UTF-8 encoding of this sentence is weighing 72 bytes of boring data.");

  public static final List<String> categories = Arrays.asList(
      "machine learning", "theory", "programming");

  public static final List<Set<String>> labels = new ArrayList<Set<String>>();
  static {
    labels.add(new HashSet<String>());
    labels.add(new HashSet<String>());
    labels.add(new HashSet<String>());
    labels.add(new HashSet<String>());
    labels.add(new HashSet<String>());
    labels.add(new HashSet<String>());
    labels.add(new HashSet<String>());
    labels.add(new HashSet<String>());
    labels
        .add(new HashSet<String>(Arrays.asList("machine learning", "theory")));
    labels.add(new HashSet<String>(Arrays.asList("programming")));
    labels
        .add(new HashSet<String>(Arrays.asList("machine learning", "theory")));
    labels.add(new HashSet<String>(Arrays.asList("programming")));
    labels.add(new HashSet<String>());
    labels
        .add(new HashSet<String>(Arrays.asList("machine learning", "theory")));
    labels.add(new HashSet<String>(Arrays.asList("machine learning",
        "programming")));
    labels.add(new HashSet<String>());
  }

  @BeforeClass
  public static void initRng() {
    RandomUtils.useTestSeed();
  }

  @Test
  public void testThresholdClassifier() {
    int numFeatures = 1000;

    OnlineLogisticRegression model = new OnlineLogisticRegression(categories
        .size() + 1, numFeatures, new L1()).lambda(0.01).learningRate(0.01);
    model.setRandomizer(new BinaryRandomizer(2, numFeatures));
    ThresholdClassifier classifier = new ThresholdClassifier(model, categories);

    // compute F1 measure on untrained model
    classifier.resetEvaluation();
    for (int i = 0; i < documents.size(); i++) {
      classifier.evaluate(documents.get(i), labels.get(i));
    }
    // the default precision is poor because of Annette
    assertEquals(0.29, classifier.getCurrentEvaluation().meanPrecision, 0.01);
    // the recall is somewhat higher since the default threshold are low enough
    assertEquals(0.72, classifier.getCurrentEvaluation().meanRecall, 0.01);
    // the F1 score is low because of the poor random precision
    assertEquals(0.40, classifier.getCurrentEvaluation().meanF1Score, 0.01);

    // train on dataset scanning the data several consecutive time to ensure
    // convergence
    for (int e = 1; e <= 20; e++) {
      for (int i = 0; i < documents.size(); i++) {
        classifier.train(documents.get(i), labels.get(i));
      }
    }

    // the F1 measure has increased to a completely fitted model for this simple
    // toy problem
    classifier.resetEvaluation();
    for (int i = 0; i < documents.size(); i++) {
      classifier.evaluate(documents.get(i), labels.get(i));
    }
    assertEquals(1.0, classifier.getCurrentEvaluation().meanF1Score, 0.001);
  }
}

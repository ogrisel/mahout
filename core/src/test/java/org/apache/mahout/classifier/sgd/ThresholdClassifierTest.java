package org.apache.mahout.classifier.sgd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.mahout.common.RandomUtils;
import org.junit.BeforeClass;
import org.junit.Test;

public class ThresholdClassifierTest {

  public static final List<String> documents = Arrays
      .asList(
          "Online learning is incremental machine learning: instances are scanned"
              + " through and the algorithm update the parameters on instance at time.",
          "Clojure is a Open Source functional programming language with Lisp"
              + " syntax that compiles to Java bytecode and run on the JVM",
          "Batch learning as opposed to online learning is a class of machine learning"
              + " algorithms that updates the parameters after each complete scan of the dataset",
          "Apache Hadoop is an Open Source Java library that implements a distributed"
              + " filesystem and a MapReduce runtime for scalable data processing.",
          "This sentences is not very interesting",
          "Clustering is an unsupervised machine learning task: data instances do not carry labels",
          "Apache Mahout is an Open Source Machine learning library written in Java that"
              + " leverages the Apache Hadoop project for scalable data processing.",
          "The UTF-8 encoding of this sentence is 53. bytes long.");

  public static final List<String> categories = Arrays.asList(
      "machine learning", "theory", "online learning", "batch learning",
      "unsupervised learning", "programming", "java", "open source");

  public static final List<List<String>> labels = new ArrayList<List<String>>();
  static {
    labels.add(Arrays.asList("machine learning", "theory", "online learning"));
    labels.add(Arrays.asList("programming", "java", "open source"));
    labels.add(Arrays.asList("machine learning", "theory", "batch learning"));
    labels.add(Arrays.asList("programming", "java", "open source"));
    labels.add(new ArrayList<String>());
    labels.add(Arrays.asList("machine learning", "theory",
        "unsupervised learning"));
    labels.add(Arrays.asList("machine learning", "programming", "java",
        "open source"));
    labels.add(new ArrayList<String>());
  }

  @BeforeClass
  public static void initRng() {
    RandomUtils.useTestSeed();
  }

  @Test
  public void testThresholdClassifier() {
    int numFeatures = 1000;

    OnlineLogisticRegression model = new OnlineLogisticRegression(categories
        .size() + 1, numFeatures, new L1()).lambda(0.01).learningRate(0.1);
    model.setRandomizer(new BinaryRandomizer(2, numFeatures));
    ThresholdClassifier classifier = new ThresholdClassifier(model, categories);

    // TODO: compute F1 measure on untrained model

    // TODO: train on dataset

    // TODO: check convergence to high F1 measure
  }

}

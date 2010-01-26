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

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Wrapper to categorize a document if the probability of a category is higher
 * than a given value. If no probability thresholds are given they are all set
 * to 0.5 by default.
 *
 * The categories probabilities are assumed to be independents. With this scheme
 * a single document can be assigned 0, 1 or several categories.
 */
public class ThresholdClassifier {

  public static final float DEFAULT_LEARNING_RATE = 0.01f;
  public static final float DEFAULT_LAMBDA = 0.01f;
  public static final int DEFAULT_WINDOW = 2; // bigrams
  public static final int DEFAULT_NUM_FEATURES = 131072; // 2 ** 17
  public static final int DEFAULT_NUM_PROBES = 2;
  public static final boolean DEFAULT_ALL_PAIRS = false;
  private static final float DEFAULT_ALPHA = 1.0f - 1e-5f;

  private final OnlineLogisticRegression[] models;
  private final List<String> allCategories;
  private Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_CURRENT,
      Collections.emptySet());
  private boolean allPairs = false;
  private int window = 2;

  private final double[] thresholds;
  private final long[] truePositiveCount;
  private final long[] falsePositiveCount;
  private final long[] falseNegativeCount;
  private final DenseVector precomputedProbs;
  private final long[] negativeSupport;
  private final long[] positiveSupport;

  public ThresholdClassifier(OnlineLogisticRegression[] models) {
    this(models, null, null);
  }

  public ThresholdClassifier(OnlineLogisticRegression[] models,
      List<String> categories) {
    this(models, categories, null);
  }

  public ThresholdClassifier(OnlineLogisticRegression[] models,
      List<String> categories, double[] thresholds) {
    this.models = models;
    for (OnlineLogisticRegression model : models) {
      if (model.getBeta().numRows() != 1) {
        throw new IllegalArgumentException(String.format(
            "Classifier expects models with a single row, got %d", model
                .getBeta().numRows()));
      }
    }
    if (categories == null) {
      categories = new ArrayList<String>(models.length);
      for (int i = 1; i <= models.length; i++) {
        categories.add(String.format("Category #", i));
      }
    } else {
      if (categories.size() != models.length) {
        throw new IllegalArgumentException(String.format(
            "Classifier need %d models, got %d", categories.size(),
            models.length));
      }
    }
    this.allCategories = categories;
    int numCategories = categories.size();
    if (thresholds == null) {
      this.thresholds = new double[numCategories];
      Arrays.fill(this.thresholds, 0.5);
    } else {
      this.thresholds = thresholds;
    }
    precomputedProbs = new DenseVector(numCategories);
    truePositiveCount = new long[numCategories];
    falsePositiveCount = new long[numCategories];
    falseNegativeCount = new long[numCategories];
    resetEvaluation();
    negativeSupport = new long[numCategories];
    positiveSupport = new long[numCategories];
  }

  /**
   * High level train method to work with category names directly. When training
   * on real dataset it is better to directly use the OnlineLogisticRegression
   * API directly to avoid overhead.
   *
   * @param document
   *          the string content of the labeled document
   * @param expectedCategories
   *          the category names of the document
   */
  public void train(String document, Set<String> categories) {
    // all models are assumed to share the same randomizer
    Vector instance = models[0].getRandomizer().randomizedInstance(
        extractTerms(document), window, allPairs);
    for (int i = 0; i < models.length; i++) {
      if (categories.contains(allCategories.get(i))) {
        models[i].train(1, instance);
      } else {
        models[i].train(0, instance);
      }
    }
  }

  /**
   * Train the models for a given instance.
   *
   * @param instance
   *          a vector representation of the instance
   * @param labels
   *          the sorted list of category indices for the instance
   */
  public void train(Vector instance, int[] labels) {
    for (int i = 0; i < models.length; i++) {
      if (Arrays.binarySearch(labels, i) != -1) {
        models[i].train(1, instance);
      } else {
        models[i].train(0, instance);
      }
    }
  }

  private void trainPrecomputed(Vector instance, int[] labels) {
    for (int i = 0; i < models.length; i++) {
      if (Arrays.binarySearch(labels, i) != -1) {
        positiveSupport[i]++;
        models[i].train(1, instance, precomputedProbs.viewPart(i, 1));
      } else {
        if (positiveSupport[i] >= negativeSupport[i]) {
          // avoid being overwhelmed by only negative examples
          negativeSupport[i]++;
          models[i].train(0, instance, precomputedProbs.viewPart(i, 1));
        }
      }
    }
  }

  /**
   * Multi-label classification of document.
   *
   * @param document
   *          the string content of the document to be tokenized and classified
   * @return the possibly empty list of matching category names
   */
  public Set<String> classify(String document) {
    // all models are assumed to share the same randomizer
    Vector instance = models[0].getRandomizer().randomizedInstance(
        extractTerms(document), window, allPairs);
    Set<String> documentCategories = new LinkedHashSet<String>();
    for (int i = 0; i < allCategories.size(); i++) {
      double prob = models[i].classify(instance).get(0);
      if (prob > thresholds[i]) {
        documentCategories.add(allCategories.get(i));
      }
    }
    return documentCategories;
  }

  public List<String> extractTerms(String document) {
    TokenStream stream = analyzer.tokenStream(null, new StringReader(document));
    TermAttribute termAtt = (TermAttribute) stream
        .addAttribute(TermAttribute.class);
    List<String> terms = new ArrayList<String>();
    try {
      while (stream.incrementToken()) {
        terms.add(termAtt.term());
      }
    } catch (IOException e) {
      // will never be raised by a StringReader
      throw new IllegalStateException(e);
    }
    return terms;
  }

  /**
   * Compute the scores (precision, recall, f1 measure) based on the current
   * count values of true positive, false positive and false negative
   * accumulated by previous call to @see #evaluate(Vector, int[]) or @see
   * #evaluate(String, Set).
   */
  public MultiLabelScores getCurrentEvaluation() {
    return new MultiLabelScores(allCategories, truePositiveCount,
        falsePositiveCount, falseNegativeCount);
  }

  /**
   * Update the evaluation counts to be able to compute F1, precision and recall
   * by later calling @see #getCurrentEvaluation()
   *
   * @param instance
   *          a vectorized instance to classify
   * @param expectedLabels
   *          the expected category indices for this instance
   */
  public void evaluate(Vector instance, int[] expectedLabels) {
    for (int i = 0; i < thresholds.length; i++) {
      double prob = models[i].classify(instance).get(0);
      if (prob > thresholds[i]) {
        if (ArrayUtils.contains(expectedLabels, i)) {
          truePositiveCount[i] += 1;
        } else {
          falsePositiveCount[i] += 1;
        }
      } else {
        if (ArrayUtils.contains(expectedLabels, i)) {
          falseNegativeCount[i] += 1;
        }
      }
      precomputedProbs.setQuick(i, prob);
    }
  }

  /**
   * Combine a performance measure followed by a train step so as to implement
   * Progressive Validation with minimum computational overhead (the
   * classification step is shared).
   *
   * Warning: this method is not thread safe because of the updates to the
   * precomputedProbs shared fields.
   *
   * @param instance
   *          a vectorized instance to classify
   * @param expectedLabels
   *          the expected category indices for this instance
   */
  public void evaluateAndTrain(Vector instance, int[] expectedLabels) {
    evaluate(instance, expectedLabels);
    trainPrecomputed(instance, expectedLabels);
  }

  /**
   * Update the evaluation counts to be able to compute F1, precision and recall
   * by later calling @see #getCurrentEvaluation()
   *
   * @param document
   *          a document to classify
   * @param expectedLabels
   *          the expected category names for this document
   */
  public void evaluate(String document, Set<String> expectedLabels) {
    Set<String> actualLabels = classify(document);
    int i = 0;
    for (String category : allCategories) {
      if (actualLabels.contains(category)) {
        if (expectedLabels.contains(category)) {
          truePositiveCount[i] += 1;
        } else {
          falsePositiveCount[i] += 1;
        }
      } else {
        if (expectedLabels.contains(category)) {
          falseNegativeCount[i] += 1;
        }
      }
      i++;
    }
  }

  /**
   * Reset the current evaluation counts.
   *
   * @see getCurrentEvaluation()
   * @see evaluate(String document, Set<String> labels)
   * @see evaluate(Vector instsance, int[] labels)
   */
  public void resetEvaluation() {
    Arrays.fill(truePositiveCount, 0);
    Arrays.fill(falsePositiveCount, 0);
    Arrays.fill(falseNegativeCount, 0);
  }

  /**
   * @return the ratio of non-zero parameters in the model
   */
  public double density() {
    double count = 0.0;
    // all models are assumed to have a single row
    for (OnlineLogisticRegression model : models) {
      count += model.getBeta().getRow(0).norm(0.0);
    }
    Matrix beta = models[0].getBeta();
    return count / (beta.numCols() * models.length);
  }

  public boolean isAllPairs() {
    return allPairs;
  }

  public void setAllPairs(boolean allPairs) {
    this.allPairs = allPairs;
  }

  public int getWindow() {
    return window;
  }

  public void setWindow(int window) {
    this.window = window;
  }

  public List<String> getCategories() {
    return allCategories;
  }

  public void setAnalyzer(Analyzer analyzer) {
    this.analyzer = analyzer;
  }

  public Analyzer getAnalyzer() {
    return analyzer;
  }

  public OnlineLogisticRegression[] getModels() {
    return models;
  }

  public static ThresholdClassifier getInstance(Configuration conf)
      throws ClassNotFoundException, InstantiationException,
      IllegalAccessException {
    // seed the RNG used initialize the model parameters
    Integer seed = conf.getInt("online.random.seed", 42);
    Random rng = RandomUtils.getRandom(seed);

    // load the list of category labels to look for
    String categoriesParamValue = conf.get("wikipedia.categories", "");
    List<String> categories = new ArrayList<String>();
    for (String category : categoriesParamValue.split(",")) {
      categories.add(category.toLowerCase().trim());
    }

    // load the randomizer that is used to hash the term of the document
    int probes = conf.getInt("randomizer.probes", DEFAULT_NUM_PROBES);
    int numFeatures = conf.getInt("randomizer.numFeatures",
        DEFAULT_NUM_FEATURES);
    TermRandomizer randomizer = new BinaryRandomizer(probes, numFeatures);
    boolean allPairs = conf
        .getBoolean("randomizer.allPairs", DEFAULT_ALL_PAIRS);
    int window = conf.getInt("randomizer.window", DEFAULT_WINDOW);

    // online learning parameters
    double lambda = conf.getFloat("online.lambda", DEFAULT_LAMBDA);
    double learningRate = conf.getFloat("online.learningRate",
        DEFAULT_LEARNING_RATE);
    double alpha = conf.getFloat("online.alpha", DEFAULT_ALPHA);

    Class<? extends PriorFunction> prior = Class.forName(
        conf.get("online.priorClass", "org.apache.mahout.classifier.sgd.L1"))
        .asSubclass(PriorFunction.class);
    double[] thresholds = null;
    if (conf.get("classifier.threshold") != null) {
      double threshold = conf.getFloat("classifier.threshold", 0.0f);
      thresholds = new double[categories.size()];
      Arrays.fill(thresholds, threshold);
    }

    // build an array of identical binary classifiers
    OnlineLogisticRegression[] models = new OnlineLogisticRegression[categories
        .size()];
    for (int i = 0; i < categories.size(); i++) {
      models[i] = new OnlineLogisticRegression(2, numFeatures, prior
          .newInstance(), rng).lambda(lambda).learningRate(learningRate).alpha(
          alpha);
      models[i].setRandomizer(randomizer);
    }
    ThresholdClassifier classifier = new ThresholdClassifier(models,
        categories, thresholds);
    classifier.setAllPairs(allPairs);
    classifier.setWindow(window);
    return classifier;
  }

}

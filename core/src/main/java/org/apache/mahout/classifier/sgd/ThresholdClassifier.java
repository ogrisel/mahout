package org.apache.mahout.classifier.sgd;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang.ArrayUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Wrapper to categorize a document if the probability of a category is higher
 * than a given value. If no probability thresholds are given, they are set to
 * (1 / (numCategories + 1).
 *
 * When classifying a document, the underlying model returns the probability for
 * the no-category case as the first component of the probabilities vector.
 *
 * With this scheme a single document can be assigned 0, 1 or several
 * categories.
 */
public class ThresholdClassifier {

  private final OnlineLogisticRegression model;
  private final List<String> allCategories;
  private Analyzer analyzer = new WhitespaceAnalyzer();
  private boolean allPairs = false;
  private int window = 2;

  private final double[] thresholds;
  private final long[] truePositiveCount;
  private final long[] falsePositiveCount;
  private final long[] falseNegativeCount;

  public ThresholdClassifier(OnlineLogisticRegression model) {
    this(model, null, null);
  }

  public ThresholdClassifier(OnlineLogisticRegression model,
      List<String> categories) {
    this(model, categories, null);
  }

  public ThresholdClassifier(OnlineLogisticRegression model,
      List<String> categories, double[] thresholds) {
    this.model = model;
    int numCategoriesAcceptedByModel = model.getBeta().numRows();
    if (categories == null) {
      categories = new ArrayList<String>(numCategoriesAcceptedByModel);
      for (int i = 1; i <= numCategoriesAcceptedByModel; i++) {
        categories.add(String.format("Category #", i));
      }
    }
    this.allCategories = categories;
    int numCategories = categories.size();
    if (numCategories != numCategoriesAcceptedByModel) {
      throw new IllegalArgumentException(String.format(
          "Model %s accepts %d categories, not %d", model,
          numCategoriesAcceptedByModel, numCategories));
    }
    if (thresholds == null) {
      double threshold = 1.0 / (numCategories + 1);
      this.thresholds = new double[numCategories];
      for (int i = 0; i < numCategories; i++) {
        this.thresholds[i] = threshold;
      }
    } else {
      this.thresholds = thresholds;
    }
    truePositiveCount = new long[numCategories];
    falsePositiveCount = new long[numCategories];
    falseNegativeCount = new long[numCategories];
    resetEvaluation();
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
  public void train(String document, Collection<String> categories) {
    Vector instance = model.getRandomizer().randomizedInstance(
        extractTerms(document), window, allPairs);
    if (categories.isEmpty()) {
      // use special category index 0 for purely negative examples
      model.train(0, instance);
    } else {
      for (String c : categories) {
        model.train(allCategories.indexOf(c) + 1, instance);
      }
    }
  }

  public void train(Vector instance, int[] labels) {
    if (labels.length == 0) {
      // use special category index 0 for purely negative examples
      model.train(0, instance);
    } else {
      for (int label : labels) {
        model.train(label + 1, instance);
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
    Vector probabilities = model.classify(extractTerms(document), window,
        allPairs);
    Set<String> documentCategories = new LinkedHashSet<String>();
    for (int i = 0; i < allCategories.size(); i++) {
      if (probabilities.get(i) > thresholds[i]) {
        documentCategories.add(allCategories.get(i));
      }
    }
    return documentCategories;
  }

  private List<String> extractTerms(String document) {
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
    Vector probabilities = model.classify(instance);
    for (int i = 0; i < thresholds.length; i++) {
      if (probabilities.get(i) > thresholds[i]) {
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
    }
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
    Matrix beta = model.getBeta();
    for (int row = 0; row < beta.numRows(); row++) {
      count += beta.getRow(row).norm(0.0);
    }
    return count / (beta.numCols() * beta.numRows());
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
}

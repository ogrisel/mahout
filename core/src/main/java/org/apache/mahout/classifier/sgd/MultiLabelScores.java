package org.apache.mahout.classifier.sgd;

import java.util.List;

/**
 * Simple datastructure to handle performance measure for the multi label
 * classification problem.
 *
 * http://en.wikipedia.org/wiki/Precision_and_recall
 */
public class MultiLabelScores implements
    Comparable<MultiLabelScores> {

  public final double meanF1Score;

  public final double meanPrecision;

  public final double meanRecall;

  public final double[] f1Score;

  public final double[] precision;

  public final double[] recall;

  private final List<String> categories; // TODO: use me in toString

  public MultiLabelScores(List<String> categories, long[] tp,
      long[] fp, long[] fn) {
    this.categories = categories;
    final int numCategories = this.categories.size();
    precision = new double[numCategories];
    recall = new double[numCategories];
    f1Score = new double[numCategories];
    double precisionSum = 0.0;
    double recallSum = 0.0;
    double f1ScoreSum = 0.0;
    for (int i = 0; i < numCategories; i++) {
      long tpAndFp = tp[i] + fp[i];
      precision[i] = tpAndFp > 0 ? ((double) tp[i]) / tpAndFp : 0;

      long tpAndFn = tp[i] + fn[i];
      recall[i] = tpAndFn > 0 ? ((double) tp[i]) / tpAndFn : 0;

      double pAndR = precision[i] + recall[i];
      f1Score[i] = pAndR > 0 ? 2.0 * precision[i] * recall[i] / pAndR : 0;

      precisionSum += precision[i];
      recallSum += recall[i];
      f1ScoreSum += f1Score[i];
    }
    meanPrecision = precisionSum / numCategories;
    meanRecall = recallSum / numCategories;
    meanF1Score = f1ScoreSum / numCategories;
  }

  @Override
  public String toString() {
    // TODO: format detailed report by category
    return String.format("precision: %02f, recall %02f, f1: %02f",
        meanPrecision, meanRecall, meanF1Score);
  }

  @Override
  public int compareTo(MultiLabelScores other) {
    return Double.compare(meanF1Score, other.meanF1Score);
  }

}

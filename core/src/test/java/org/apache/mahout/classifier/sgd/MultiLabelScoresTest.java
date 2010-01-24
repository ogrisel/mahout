package org.apache.mahout.classifier.sgd;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class MultiLabelScoresTest extends Assert {

  @Test
  public void testNullScores() throws Exception {
    List<String> categories = Arrays.asList("c1", "c2", "c3");
    long[] zeroCounts = {0, 0, 0};
    MultiLabelScores scores = new MultiLabelScores(categories, zeroCounts,
        zeroCounts, zeroCounts);

    assertEquals(0.0, scores.meanF1Score, 0.0);
    assertEquals(0.0, scores.meanPrecision, 0.0);
    assertEquals(0.0, scores.meanRecall, 0.0);

    double[] zeroRates = {0, 0, 0};
    assertArrayEquals(zeroRates, scores.f1Score, 0.0);
    assertArrayEquals(zeroRates, scores.precision, 0.0);
    assertArrayEquals(zeroRates, scores.recall, 0.0);
  }

  @Test
  public void testRegularScores() throws Exception {
    List<String> categories = Arrays.asList("c1", "c2", "c3");
    long[] tp = {1013, 4201, 11};
    long[] fp = {80, 512, 0};
    long[] fn = {212, 612, 3};
    MultiLabelScores scores = new MultiLabelScores(categories, tp, fp, fn);

    assertEquals(0.88, scores.meanF1Score, 0.01);
    assertEquals(0.94, scores.meanPrecision, 0.01);
    assertEquals(0.83, scores.meanRecall, 0.01);

    assertArrayEquals(new double[] {0.87, 0.88, 0.88}, scores.f1Score, 0.01);
    assertArrayEquals(new double[] {0.92, 0.89, 1.00}, scores.precision, 0.01);
    assertArrayEquals(new double[] {0.83, 0.87, 0.79}, scores.recall, 0.01);
  }

}

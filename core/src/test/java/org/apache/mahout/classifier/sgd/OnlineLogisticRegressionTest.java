package org.apache.mahout.classifier.sgd;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Assert;
import org.junit.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

public class OnlineLogisticRegressionTest {
  @Test
  public void testClassify() throws Exception {
    OnlineLogisticRegression lr = new OnlineLogisticRegression(3, 2, new L2(1));
    // set up some internal coefficients as if we had learned them
    lr.setBeta(0, 0, -1);
    lr.setBeta(1, 0, -2);

    // zero vector gives no information.  All classes are equal.
    Vector v = lr.classify(new DenseVector(new double[]{0, 0}));
    Assert.assertEquals(1 / 3.0, v.get(0), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(1), 1e-8);

    v = lr.classifyFullVector(new DenseVector(new double[]{0, 0}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(0), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(1), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(2), 1e-8);

    // weights for second vector component are still zero
    v = lr.classify(new DenseVector(new double[]{0, 1}));
    Assert.assertEquals(1 / 3.0, v.get(0), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(1), 1e-8);

    v = lr.classifyFullVector(new DenseVector(new double[]{0, 1}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(0), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(1), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(2), 1e-8);

    // but the weights on the first component are non-zero
    v = lr.classify(new DenseVector(new double[]{1, 0}));
    Assert.assertEquals(Math.exp(-1) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(0), 1e-8);
    Assert.assertEquals(Math.exp(-2) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(1), 1e-8);

    v = lr.classifyFullVector(new DenseVector(new double[]{1, 0}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(Math.exp(-1) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(0), 1e-8);
    Assert.assertEquals(Math.exp(-2) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(1), 1e-8);
    Assert.assertEquals(1 / (1 + Math.exp(-1) + Math.exp(-2)), v.get(2), 1e-8);

    lr.setBeta(0, 1, 1);

    v = lr.classifyFullVector(new DenseVector(new double[]{1, 1}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(Math.exp(0) / (1 + Math.exp(0) + Math.exp(-2)), v.get(0), 1e-8);
    Assert.assertEquals(Math.exp(-2) / (1 + Math.exp(0) + Math.exp(-2)), v.get(1), 1e-8);
    Assert.assertEquals(1 / (1 + Math.exp(0) + Math.exp(-2)), v.get(2), 1e-8);

    lr.setBeta(1, 1, 3);

    v = lr.classifyFullVector(new DenseVector(new double[]{1, 1}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(Math.exp(0) / (1 + Math.exp(0) + Math.exp(1)), v.get(0), 1e-8);
    Assert.assertEquals(Math.exp(1) / (1 + Math.exp(0) + Math.exp(1)), v.get(1), 1e-8);
    Assert.assertEquals(1 / (1 + Math.exp(0) + Math.exp(1)), v.get(2), 1e-8);
  }

  @Test
  public void testTrain() throws Exception {
    Matrix input = readCsv("/sgd.csv");
    Matrix results = readCsv("/r.csv");
    Matrix logP = readCsv("/logP.csv");

    OnlineLogisticRegression lr10 = new OnlineLogisticRegression(2, 8, new L1()).lambda(10).learningRate(0.1);
    OnlineLogisticRegression lr1 = new OnlineLogisticRegression(2, 8, new L1()).lambda(1).learningRate(0.1);
    OnlineLogisticRegression lr01 = new OnlineLogisticRegression(2, 8, new L1()).lambda(0.1).learningRate(0.1);

    RandomUtils.useTestSeed();
    Random gen = RandomUtils.getRandom();
    int[] permutation = new int[60];
    permutation[0] = 0;
    for (int i = 1; i < 60; i++) {
      int n = gen.nextInt(i + 1);
      if (n != i) {
        permutation[i] = permutation[n];
        permutation[n] = i;
      } else {
        permutation[i] = i;
      }
    }

    for (int i = 0; i < 20; i++) {
      for (int row : permutation) {
        // why do lr10 and lr1 converge to zero while lr01 does the right thing?
        lr10.train(row / (input.numRows() / 2), input.getRow(row));
        lr1.train(row / (input.numRows() / 2), input.getRow(row));
        lr01.train(row / (input.numRows() / 2), input.getRow(row));
      }
    }
  }

  private Matrix readCsv(String resourceName) {
    InputStream is = this.getClass().getResourceAsStream(resourceName);
    InputStreamReader isr = new InputStreamReader(is);
    Scanner s = new Scanner(isr);
    s.useDelimiter("\n");

    Map<String, Integer> labels = new HashMap<String, Integer>();
    int column = 0;
    for (String label : s.next().split(",")) {
      labels.put(label, column);
    }

    List<String> data = new ArrayList<String>();
    while (s.hasNext()) {
      data.add(s.next());
    }

    Matrix r = new DenseMatrix(data.size(), data.get(0).split(",").length);
    r.setRowLabelBindings(labels);
    int i = 0;
    for (String line : data) {
      int j = 0;
      for (String value : line.split(",")) {
        r.set(i, j, Double.parseDouble(value));
        j++;
      }
      i++;
    }

    return r;
  }
}

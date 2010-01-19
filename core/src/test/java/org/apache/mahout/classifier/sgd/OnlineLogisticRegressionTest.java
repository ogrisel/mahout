package org.apache.mahout.classifier.sgd;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class OnlineLogisticRegressionTest {

  @Before
  public void initRng() {
    RandomUtils.useTestSeed();
  }

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
    Assert.assertEquals(1 / 3.0, v.get(0), 1e-3);
    Assert.assertEquals(1 / 3.0, v.get(1), 1e-3);

    v = lr.classifyFullVector(new DenseVector(new double[]{0, 1}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(1 / 3.0, v.get(0), 1e-3);
    Assert.assertEquals(1 / 3.0, v.get(1), 1e-3);
    Assert.assertEquals(1 / 3.0, v.get(2), 1e-3);

    // but the weights on the first component are non-zero
    v = lr.classify(new DenseVector(new double[]{1, 0}));
    Assert.assertEquals(Math.exp(-1) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(0), 1e-8);
    Assert.assertEquals(Math.exp(-2) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(1), 1e-8);

    v = lr.classifyFullVector(new DenseVector(new double[]{1, 0}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(1 / (1 + Math.exp(-1) + Math.exp(-2)), v.get(0), 1e-8);
    Assert.assertEquals(Math.exp(-1) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(1), 1e-8);
    Assert.assertEquals(Math.exp(-2) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(2), 1e-8);

    lr.setBeta(0, 1, 1);

    v = lr.classifyFullVector(new DenseVector(new double[]{1, 1}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(1 / (1 + Math.exp(0) + Math.exp(-2)), v.get(0), 1e-3);
    Assert.assertEquals(Math.exp(0) / (1 + Math.exp(0) + Math.exp(-2)), v.get(1), 1e-3);
    Assert.assertEquals(Math.exp(-2) / (1 + Math.exp(0) + Math.exp(-2)), v.get(2), 1e-3);

    lr.setBeta(1, 1, 3);

    v = lr.classifyFullVector(new DenseVector(new double[]{1, 1}));
    Assert.assertEquals(1.0, v.zSum(), 1e-8);
    Assert.assertEquals(1 / (1 + Math.exp(0) + Math.exp(1)), v.get(0), 1e-8);
    Assert.assertEquals(Math.exp(0) / (1 + Math.exp(0) + Math.exp(1)), v.get(1), 1e-8);
    Assert.assertEquals(Math.exp(1) / (1 + Math.exp(0) + Math.exp(1)), v.get(2), 1e-8);
  }

  @Test
  public void testTrain() throws Exception {
    Matrix input = readCsv("/sgd.csv");

    OnlineLogisticRegression lr10 = new OnlineLogisticRegression(2, 8, new L1()).lambda(10).learningRate(0.1);
    OnlineLogisticRegression lr1 = new OnlineLogisticRegression(2, 8, new L1()).lambda(1).learningRate(0.1);
    OnlineLogisticRegression lr01 = new OnlineLogisticRegression(2, 8, new L1()).lambda(0.1).learningRate(0.1);

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

  @Test
  public void testTrainingConvergenceToReference() throws Exception {
    // build a linearly separable training set by using a randomly initialized
    // reference model used to classify a random set of previously unlabeled
    // input vectors

    int numCategories = 3;
    int numFeatures = 420;
    int trainingSetSize = 100;
    Random rng = RandomUtils.getRandom(0);
    OnlineLogisticRegression reference = new OnlineLogisticRegression(
        numCategories, numFeatures, new UniformPrior(), rng);

    Matrix input = randomDenseMatrix(trainingSetSize, numFeatures, 1, rng);
    int[] labels = new int[trainingSetSize];
    for (int i = 0; i < trainingSetSize; i++) {
      labels[i] = reference.classifyFullVector(input.getRow(i)).maxValueIndex();
    }

    // init a batch of new models with random parameters and train it with the
    // reference dataset
    double learningRate = 0.1;
    double lambda = 0.005;

    OnlineLogisticRegression lrL1 = new OnlineLogisticRegression(numCategories,
        numFeatures, new L1()).lambda(lambda).learningRate(learningRate);
    OnlineLogisticRegression lrL2 = new OnlineLogisticRegression(numCategories,
        numFeatures, new L2(1.0)).lambda(lambda).learningRate(learningRate);
    OnlineLogisticRegression lrU = new OnlineLogisticRegression(numCategories,
        numFeatures, new UniformPrior()).lambda(lambda).learningRate(
        learningRate);
    OnlineLogisticRegression lrT = new OnlineLogisticRegression(numCategories,
        numFeatures, new TPrior(1.0)).lambda(lambda).learningRate(learningRate);

    List<OnlineLogisticRegression> models = Arrays.asList(lrL1, lrL2, lrU, lrT);

    int epochs = 10;

    for (OnlineLogisticRegression model : models) {
      // untrained model has a high error rate
      double untrainedErrorRate = misClassificationRate(model, input, labels);
      double expectedMinRate = 0.5;
      Assert.assertTrue(String.format(
          "misclassification rate '%f' is lower than expected '%f'",
          untrainedErrorRate, expectedMinRate),
          untrainedErrorRate > expectedMinRate);

      // check that untrained model is dense
      double untrainedDensity = density(model);
      double minExpectedDensity = 0.99;
      Assert.assertTrue(String.format(
          "model density '%f' is lower than expected '%f'",
          untrainedDensity, minExpectedDensity),
          untrainedDensity > minExpectedDensity);

      // train the model
      for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < trainingSetSize; i++) {
          model.train(labels[i], input.getRow(i));
        }
      }

      // check the convergence of the parameters towards accurate models
      double trainedErrorRate = misClassificationRate(model, input, labels);
      double expectedMaxRate = 0.01;
      Assert.assertTrue(String.format(
          "misclassification rate '%f' is larger than expected '%f'",
          trainedErrorRate, expectedMaxRate),
          trainedErrorRate < expectedMaxRate);

      if (model.getPrior().isSparsityInducing()) {
        // check that model trained with a L1 Prior is sparse according to the
        // regularization strength lambda
        // NB. this example use a dataset that was generated from a dense
        // reference model hence one cannot expect an important sparsity without
        // loosing much in accuracy
        double actualDensity = density(model);
        double maxExpectedDensity = 0.85;
        Assert.assertTrue(String.format(
            "model density '%f' is larger than expected '%f'",
            actualDensity, maxExpectedDensity),
            actualDensity < maxExpectedDensity);
      } else {
        // check that model trained with non-L1 Prior is dense whatever the
        // regularization strength lambda
        double actualDensity = density(model);
        Assert.assertTrue(String.format(
            "model density '%f' is lower than expected '%f'",
            actualDensity, minExpectedDensity),
            actualDensity > minExpectedDensity);
      }
    }
  }

  private double density(OnlineLogisticRegression model) {
    double count = 0.0;
    Matrix beta = model.getBeta();
    for (int row = 0; row < beta.numRows(); row++) {
      count += beta.getRow(row).norm(0.0);
    }
    return count / (beta.numCols() * beta.numRows());
  }

  private double misClassificationRate(OnlineLogisticRegression model,
      Matrix input, int[] labels) {
    double count = 0;
    for (int i = 0; i < labels.length; i++) {
      Vector classification = model.classifyFullVector(input.getRow(i));
      if (labels[i] != classification.maxValueIndex()) {
        count++;
      }
    }
    return count / labels.length;
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

  private Matrix randomDenseMatrix(int rows, int columns, double variance,
      Random rng) {
    // TODO: factorize this utility somewhere
    DenseMatrix matrix = new DenseMatrix(rows, columns);
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        matrix.set(row, column, rng.nextGaussian() * variance);
      }
    }
    return matrix;
  }
}

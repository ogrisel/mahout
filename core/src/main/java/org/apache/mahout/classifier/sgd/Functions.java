package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.UnaryFunction;


/**
 * Some handy functions.
 * TODO move this into the old Colt structure.  This is only separate because Mahout's UnaryFunction is required
 * for matrix operations, but the old Colt code provides DoubleFunctions.
 */
public class Functions {
  public static UnaryFunction exp = new UnaryFunction() {
    @Override
    public double apply(double x) {
      return Math.exp(x);
    }
  };
}

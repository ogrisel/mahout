package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.Vector;

/**
 * Continuous values are stored in fixed randomized location in the feature vector.
 */
public class ContinuousValueEncoder extends RecordValueEncoder {
  public ContinuousValueEncoder(String name) {
    super(name);
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, Vector data) {
    for (int i = 0; i < probes; i++) {
      int n = hash(name, CONTINUOUS_VALUE_HASH_SEED + i, data.size());
      trace(name, null, n);
      data.set(n, data.get(n) + Double.parseDouble(originalForm));
    }
  }

  /**
   * Converts a value into a form that would help a human understand the internals of how the value
   * is being interpreted.  For text-like things, this is likely to be a list of the terms found with
   * associated weights (if any).
   *
   * @param originalForm The original form of the value as a string.
   * @return A string that a human can read.
   */
  @Override
  public String asString(String originalForm) {
    return name + ":" + originalForm;
  }
}

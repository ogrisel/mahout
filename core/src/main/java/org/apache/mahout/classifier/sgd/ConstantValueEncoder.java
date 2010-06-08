package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.Vector;

/**
 * An encoder that does the standard thing for a virtual bias term.
 */
public class ConstantValueEncoder extends RecordValueEncoder {
  public ConstantValueEncoder(String name) {
    super(name);
  }

  @Override
  public void addToVector(String originalForm, Vector data) {
    for (int i = 0; i < probes; i++) {
      int n = hash(name, i, data.size());
      trace(name, null, n);
      data.set(n, data.get(n) + 1);
    }
  }

  @Override
  public String asString(String originalForm) {
    return name;
  }
}

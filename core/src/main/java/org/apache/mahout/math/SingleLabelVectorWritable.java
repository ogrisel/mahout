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

package org.apache.mahout.math;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Writable to handle serialization of a vector and a single label index.
 */
public class SingleLabelVectorWritable extends VectorWritable {

  private int label = 0;

  public void setLabel(int label) {
    this.label = label;
  }

  public int getLabel() {
    return label;
  }

  public SingleLabelVectorWritable() {}

  public SingleLabelVectorWritable(Vector v, int label) {
    super(v);
    setLabel(label);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    label = in.readInt();
    super.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(label);
    super.write(out);
  }

  public static SingleLabelVectorWritable read(
      DataInput in) throws IOException {
    int label = in.readInt();
    Vector vector = VectorWritable.readVector(in);
    return new SingleLabelVectorWritable(vector, label);
  }

  public static void write(DataOutput out, Vector v, int label)
      throws IOException {
    (new SingleLabelVectorWritable(v, label)).write(out);
  }

}

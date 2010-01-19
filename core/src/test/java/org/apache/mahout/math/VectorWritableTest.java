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

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import junit.framework.Assert;

import org.apache.hadoop.io.DataOutputBuffer;
import org.junit.Before;
import org.junit.Test;

public class VectorWritableTest {

  DenseVector dv;
  RandomAccessSparseVector rsv;
  SequentialAccessSparseVector ssv;

  @Before
  public void makeSomeVectors() {
    dv = new DenseVector(new double[] {0.3, -1.0, 0.0, 0.0, 0.0, -42.0, 0.42,
        0.0});
    dv.setName("Nemo");
    rsv = new RandomAccessSparseVector(dv);
    ssv = new SequentialAccessSparseVector(dv);
  }

  @Test
  public void testDenseVectorWritable() throws IOException {
    DataOutputBuffer outBuffer = new DataOutputBuffer();
    VectorWritable.writeVector(outBuffer, dv);
    InputStream is = new ByteArrayInputStream(outBuffer.getData());
    Vector newVector = VectorWritable.readVector(new DataInputStream(is));
    Assert.assertTrue(newVector instanceof DenseVector);
    Assert.assertTrue(String.format(
        "deserialized vector is nor equal to serialized version: %s != %s",
        newVector.asFormatString(), dv.asFormatString()), AbstractVector
        .equivalent(dv, newVector));
    Assert.assertEquals("Nemo", newVector.getName());
  }

  @Test
  public void testRandomAccessSparseVectorWritable() throws IOException {
    DataOutputBuffer outBuffer = new DataOutputBuffer();
    VectorWritable.writeVector(outBuffer, rsv);
    InputStream is = new ByteArrayInputStream(outBuffer.getData());
    Vector newVector = VectorWritable.readVector(new DataInputStream(is));
    Assert.assertTrue(newVector instanceof RandomAccessSparseVector);
    Assert.assertTrue(String.format(
        "deserialized vector is nor equal to serialized version: %s != %s",
        newVector.asFormatString(), rsv.asFormatString()), AbstractVector
        .equivalent(rsv, newVector));
    Assert.assertEquals("Nemo", newVector.getName());
  }

  @Test
  public void testSequentialAccessSparseVectorWritable() throws IOException {
    DataOutputBuffer outBuffer = new DataOutputBuffer();
    VectorWritable.writeVector(outBuffer, ssv);
    InputStream is = new ByteArrayInputStream(outBuffer.getData());
    Vector newVector = VectorWritable.readVector(new DataInputStream(is));
    Assert.assertTrue(newVector instanceof SequentialAccessSparseVector);
    Assert.assertTrue(String.format(
        "deserialized vector is nor equal to serialized version: %s != %s",
        newVector.asFormatString(), ssv.asFormatString()), AbstractVector
        .equivalent(ssv, newVector));
    Assert.assertEquals("Nemo", newVector.getName());
  }

  @Test
  public void testSingleLabelVectorWritable() throws IOException {
    DataOutputBuffer outBuffer = new DataOutputBuffer();
    SingleLabelVectorWritable.write(outBuffer, ssv, 42);
    InputStream is = new ByteArrayInputStream(outBuffer.getData());
    SingleLabelVectorWritable slvw = SingleLabelVectorWritable
        .read(new DataInputStream(is));
    Assert.assertEquals(42, slvw.getLabel());
    Vector newVector = slvw.get();
    Assert.assertTrue(newVector instanceof SequentialAccessSparseVector);
    Assert.assertTrue(String.format(
        "deserialized vector is nor equal to serialized version: %s != %s",
        newVector.asFormatString(), ssv.asFormatString()), AbstractVector
        .equivalent(ssv, newVector));
    Assert.assertEquals("Nemo", newVector.getName());
  }

  @Test
  public void testMultiLabelVectorWritable() throws IOException {
    DataOutputBuffer outBuffer = new DataOutputBuffer();
    MultiLabelVectorWritable.write(outBuffer, ssv, new int[] {42, 1});
    InputStream is = new ByteArrayInputStream(outBuffer.getData());
    MultiLabelVectorWritable mlvw = MultiLabelVectorWritable
        .read(new DataInputStream(is));
    Assert.assertTrue(Arrays.equals(new int[] {42, 1}, mlvw.getLabels()));
    Vector newVector = mlvw.get();
    Assert.assertTrue(newVector instanceof SequentialAccessSparseVector);
    Assert.assertTrue(String.format(
        "deserialized vector is nor equal to serialized version: %s != %s",
        newVector.asFormatString(), ssv.asFormatString()), AbstractVector
        .equivalent(ssv, newVector));
    Assert.assertEquals("Nemo", newVector.getName());
  }

  @Test
  public void testEmptyMultiLabelVectorWritable() throws IOException {
    DataOutputBuffer outBuffer = new DataOutputBuffer();
    MultiLabelVectorWritable.write(outBuffer, ssv, new int[] {});
    InputStream is = new ByteArrayInputStream(outBuffer.getData());
    MultiLabelVectorWritable mlvw = MultiLabelVectorWritable
        .read(new DataInputStream(is));
    Assert.assertTrue(Arrays.equals(new int[] {}, mlvw.getLabels()));
    Vector newVector = mlvw.get();
    Assert.assertTrue(newVector instanceof SequentialAccessSparseVector);
    Assert.assertTrue(String.format(
        "deserialized vector is nor equal to serialized version: %s != %s",
        newVector.asFormatString(), ssv.asFormatString()), AbstractVector
        .equivalent(ssv, newVector));
    Assert.assertEquals("Nemo", newVector.getName());
  }

}

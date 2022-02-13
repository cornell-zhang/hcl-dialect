// RUN: hcl-opt -opt %s | FileCheck %s

module {
  func @conv_interleaving_accu(%Input: memref<32x32xf32>, %Kernel: memref<3x3xf32>, %Output: memref<30x30xf32>)
  {
    %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
    %lry = hcl.create_loop_handle "ry" : !hcl.LoopHandle
    %lrx = hcl.create_loop_handle "rx" : !hcl.LoopHandle
    %s = hcl.create_stage_handle "s" : !hcl.StageHandle
    affine.for %i = 0 to 30 {
      // CHECK: %[[MEM:.*]] = memref.alloc() : memref<30xf32>
      // CHECK: %cst = arith.constant 0.000000e+00 : f32
      // CHECK: affine.for %[[VAR:.*]] = 0 to 30 {
      // CHECK:   affine.store %cst, %[[MEM]][%[[VAR]]] : memref<30xf32>
      // CHECK: } {loop_name = "j_init", pipeline_ii = 1 : i32}
      // CHECK: affine.for {{.*}} = 0 to 3 {
      affine.for %j = 0 to 30 {
        // CHECK: affine.for {{.*}} = 0 to 3 {
        affine.for %ry = 0 to 3 {
          // CHECK: affine.for %[[VAR1:.*]] = 0 to 30 {
          affine.for %rx = 0 to 3 {
            %a = affine.load %Input[%i+%ry, %j+%rx] : memref<32x32xf32>
            %k = affine.load %Kernel[%ry, %rx] : memref<3x3xf32>
            %b = affine.load %Output[%i, %j] : memref<30x30xf32>
            %mul = arith.mulf %a, %k : f32
            %sum = arith.addf %b, %mul : f32
            // CHECK: affine.store {{.*}}, %[[MEM]][%[[VAR1]]] : memref<30xf32>
            affine.store %sum, %Output[%i, %j] : memref<30x30xf32>
          } { loop_name = "rx", reduction = 1 }
        } { loop_name = "ry", reduction = 1 }
      } { loop_name = "j" }
      // CHECK: } {loop_name = "ry", reduction = 1 : i64}
      // CHECK: affine.for %[[VAR]] = 0 to 30 {
      // CHECK:   %[[RES:.*]] = affine.load %[[MEM]][%[[VAR]]] : memref<30xf32>
      // CHECK:   affine.store %[[RES]], {{.*}}[{{.*}}, %[[VAR]]] : memref<30x30xf32>
      // CHECK: } {loop_name = "j_back", pipeline_ii = 1 : i32}
    } { loop_name = "i", stage_name = "s" }
    hcl.reorder(%s, %lry, %lrx, %lj)
    %buf = hcl.buffer_at(%s, %Output: memref<30x30xf32>, %li) -> memref<30xf32>
    return
  }
  func @conv2d_interleaving_accu(%Input: memref<3x32x32xf32>, %Kernel: memref<6x3x3x3xf32>, %Output: memref<6x30x30xf32>)
  {
    %loc = hcl.create_loop_handle "oc" : !hcl.LoopHandle
    %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
    %lrc = hcl.create_loop_handle "rc" : !hcl.LoopHandle
    %lry = hcl.create_loop_handle "ry" : !hcl.LoopHandle
    %lrx = hcl.create_loop_handle "rx" : !hcl.LoopHandle
    %s = hcl.create_stage_handle "s" : !hcl.StageHandle
    affine.for %oc = 0 to 6 { // out channel
      affine.for %i = 0 to 30 {
        affine.for %j = 0 to 30 {
          affine.for %rc = 0 to 3 { // in channel
            affine.for %ry = 0 to 3 {
              affine.for %rx = 0 to 3 {
                %a = affine.load %Input[%rc, %i+%ry, %j+%rx] : memref<3x32x32xf32>
                %k = affine.load %Kernel[%oc, %rc, %ry, %rx] : memref<6x3x3x3xf32>
                %b = affine.load %Output[%oc, %i, %j] : memref<6x30x30xf32>
                %mul = arith.mulf %a, %k : f32
                %sum = arith.addf %b, %mul : f32
                affine.store %sum, %Output[%oc, %i, %j] : memref<6x30x30xf32>
              } { loop_name = "rx", reduction = 1 }
            } { loop_name = "ry", reduction = 1 }
          } { loop_name = "rc", reduction = 1 }
        } { loop_name = "j" }
      } { loop_name = "i" }
    } {loop_name = "oc", stage_name = "s" }
    hcl.reorder(%s, %lrc, %lry, %lrx, %lj)
    %buf = hcl.buffer_at(%s, %Output: memref<6x30x30xf32>, %li) -> memref<30xf32>
    return
  }
  func @conv2d_default_buf(%Input: memref<3x32x32xf32>, %Kernel: memref<6x3x3x3xf32>, %Output: memref<6x30x30xf32>)
  {
    %loc = hcl.create_loop_handle "oc" : !hcl.LoopHandle
    %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
    %lrc = hcl.create_loop_handle "rc" : !hcl.LoopHandle
    %lry = hcl.create_loop_handle "ry" : !hcl.LoopHandle
    %lrx = hcl.create_loop_handle "rx" : !hcl.LoopHandle
    %s = hcl.create_stage_handle "s" : !hcl.StageHandle
    affine.for %oc = 0 to 6 { // out channel
      affine.for %i = 0 to 30 {
        affine.for %j = 0 to 30 {
          affine.for %rc = 0 to 3 { // in channel
            affine.for %ry = 0 to 3 {
              affine.for %rx = 0 to 3 {
                %a = affine.load %Input[%rc, %i+%ry, %j+%rx] : memref<3x32x32xf32>
                %k = affine.load %Kernel[%oc, %rc, %ry, %rx] : memref<6x3x3x3xf32>
                %b = affine.load %Output[%oc, %i, %j] : memref<6x30x30xf32>
                %mul = arith.mulf %a, %k : f32
                %sum = arith.addf %b, %mul : f32
                affine.store %sum, %Output[%oc, %i, %j] : memref<6x30x30xf32>
              } { loop_name = "rx", reduction = 1 }
            } { loop_name = "ry", reduction = 1 }
          } { loop_name = "rc", reduction = 1 }
        } { loop_name = "j" }
      } { loop_name = "i" }
    } {loop_name = "oc", stage_name = "s" }
    %buf = hcl.buffer_at(%s, %Output: memref<6x30x30xf32>, %lj) -> memref<1xf32>
    return
  }
  func @conv2d_buffer_at_0(%Input: memref<3x32x32xf32>, %Kernel: memref<6x3x3x3xf32>, %Output: memref<6x30x30xf32>)
  {
    %loc = hcl.create_loop_handle "oc" : !hcl.LoopHandle
    %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
    %lrc = hcl.create_loop_handle "rc" : !hcl.LoopHandle
    %lry = hcl.create_loop_handle "ry" : !hcl.LoopHandle
    %lrx = hcl.create_loop_handle "rx" : !hcl.LoopHandle
    %s = hcl.create_stage_handle "s" : !hcl.StageHandle
    affine.for %oc = 0 to 6 { // out channel
    // CHECK: %[[MEM:.*]] = memref.alloc() : memref<30x30xf32>
    // CHECK: %cst = arith.constant 0.000000e+00 : f32
    // CHECK: affine.for %[[VAR1:.*]] = 0 to 30 {
    // CHECK:   affine.for %[[VAR2:.*]] = 0 to 30 {
    // CHECK:     affine.store %cst, %[[MEM:.*]][%[[VAR1]], %[[VAR2]]] : memref<30x30xf32>
    // CHECK:   } {pipeline_ii = 1 : i32}
    // CHECK: } {loop_name = "i_init"}
      affine.for %i = 0 to 30 {
        affine.for %j = 0 to 30 {
          affine.for %rc = 0 to 3 { // in channel
            affine.for %ry = 0 to 3 {
              affine.for %rx = 0 to 3 {
                %a = affine.load %Input[%rc, %i+%ry, %j+%rx] : memref<3x32x32xf32>
                %k = affine.load %Kernel[%oc, %rc, %ry, %rx] : memref<6x3x3x3xf32>
                %b = affine.load %Output[%oc, %i, %j] : memref<6x30x30xf32>
                %mul = arith.mulf %a, %k : f32
                %sum = arith.addf %b, %mul : f32
                // CHECK: affine.store {{.*}}, %[[MEM]][{{.*}}, {{.*}}] : memref<30x30xf32>
                affine.store %sum, %Output[%oc, %i, %j] : memref<6x30x30xf32>
              } { loop_name = "rx", reduction = 1 }
            } { loop_name = "ry", reduction = 1 }
          } { loop_name = "rc", reduction = 1 }
        } { loop_name = "j" }
      } { loop_name = "i" }
      // CHECK: affine.for %[[VAR1]] = 0 to 30 {
      // CHECK:   affine.for %[[VAR2]] = 0 to 30 {
      // CHECK:     %[[RES:.*]] = affine.load %[[MEM]][%[[VAR1]], %[[VAR2]]] : memref<30x30xf32>
      // CHECK:     affine.store %[[RES]], {{.*}}[{{.*}}, {{.*}}, {{.*}}] : memref<6x30x30xf32>
      // CHECK:   } {pipeline_ii = 1 : i32}
      // CHECK: } {loop_name = "i_back"}
    } {loop_name = "oc", stage_name = "s" }
    %buf = hcl.buffer_at(%s, %Output: memref<6x30x30xf32>, %loc) -> memref<30x30xf32>
    return
  }
  func @conv2d_buffer_at_0_interleaving(%Input: memref<3x32x32xf32>, %Kernel: memref<6x3x3x3xf32>, %Output: memref<6x30x30xf32>)
  {
    %loc = hcl.create_loop_handle "oc" : !hcl.LoopHandle
    %li = hcl.create_loop_handle "i" : !hcl.LoopHandle
    %lj = hcl.create_loop_handle "j" : !hcl.LoopHandle
    %lrc = hcl.create_loop_handle "rc" : !hcl.LoopHandle
    %lry = hcl.create_loop_handle "ry" : !hcl.LoopHandle
    %lrx = hcl.create_loop_handle "rx" : !hcl.LoopHandle
    %s = hcl.create_stage_handle "s" : !hcl.StageHandle
    affine.for %oc = 0 to 6 { // out channel
      // CHECK: %[[MEM:.*]] = memref.alloc() : memref<30x30xf32>
      // CHECK: %cst = arith.constant 0.000000e+00 : f32
      // CHECK: affine.for %[[VAR1:.*]] = 0 to 30 {
      // CHECK:   affine.for %[[VAR2:.*]] = 0 to 30 {
      // CHECK:     affine.store %cst, %[[MEM:.*]][%[[VAR1]], %[[VAR2]]] : memref<30x30xf32>
      // CHECK:   } {pipeline_ii = 1 : i32}
      // CHECK: } {loop_name = "i_init"}
      affine.for %i = 0 to 30 {
        affine.for %j = 0 to 30 {
          affine.for %rc = 0 to 3 { // in channel
            affine.for %ry = 0 to 3 {
              affine.for %rx = 0 to 3 {
                %a = affine.load %Input[%rc, %i+%ry, %j+%rx] : memref<3x32x32xf32>
                %k = affine.load %Kernel[%oc, %rc, %ry, %rx] : memref<6x3x3x3xf32>
                %b = affine.load %Output[%oc, %i, %j] : memref<6x30x30xf32>
                %mul = arith.mulf %a, %k : f32
                %sum = arith.addf %b, %mul : f32
                affine.store %sum, %Output[%oc, %i, %j] : memref<6x30x30xf32>
              } { loop_name = "rx", reduction = 1 }
            } { loop_name = "ry", reduction = 1 }
          } { loop_name = "rc", reduction = 1 }
        } { loop_name = "j" }
      } { loop_name = "i" }
      // CHECK:         } {loop_name = "j"}
      // CHECK:       } {loop_name = "i"}
      // CHECK:     } {loop_name = "rx", reduction = 1 : i64}
      // CHECK:   } {loop_name = "ry", reduction = 1 : i64}
      // CHECK: } {loop_name = "rc", reduction = 1 : i64}
      // CHECK: affine.for %[[VAR1]] = 0 to 30 {
      // CHECK:   affine.for %[[VAR2]] = 0 to 30 {
      // CHECK:     %[[RES:.*]] = affine.load %[[MEM]][%[[VAR1]], %[[VAR2]]] : memref<30x30xf32>
      // CHECK:     affine.store %[[RES]], {{.*}}[{{.*}}, {{.*}}, {{.*}}] : memref<6x30x30xf32>
      // CHECK:   } {pipeline_ii = 1 : i32}
      // CHECK: } {loop_name = "i_back"}
    } {loop_name = "oc", stage_name = "s" }
    hcl.reorder(%s, %lrc, %lry, %lrx, %li, %lj)
    %buf = hcl.buffer_at(%s, %Output: memref<6x30x30xf32>, %loc) -> memref<30x30xf32>
    return
  }
}
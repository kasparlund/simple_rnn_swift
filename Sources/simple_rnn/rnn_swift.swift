import Foundation
import TensorFlow

func rnn_swift() {
  print("Learning sherlock_holmes using a simple RNN");
  //load data and create vocabulary
  let url        = URL(fileURLWithPath: "/Users/kasparlund/AICodeData/data/sherlock/sherlock_holmes_the_canon.txt")
  let data       = try! Array<Character>(String(NSString(contentsOfFile: url.path, encoding: String.Encoding.ascii.rawValue)))
  let chars      = Set<Character> ( data )
  let data_size  : Int32 = Int32( data.count  )
  let vocab_size : Int32 = Int32( chars.count )
  print("data has \(data_size) characters with, \(vocab_size) unique characters")

  //create encoder and decoder
  var char_to_ix = [Character: Int32]()
  var ix_to_char = [Int32: Character]()
  for (i,c) in chars.enumerated() {
    char_to_ix[c]        = Int32(i)
    ix_to_char[Int32(i)] = c
    //print(c) //print character in vocabulary
  }

  // hyperparameters
  let hidden_size   : Int32  = 100      // size of hidden layer of neurons
  let seq_length    : Int32  = 25       // bptt: number of steps to unroll the RNN for
  let learning_rate : Float  = 1e-1

  var inputs  = Array<Int32>( repeating: 0, count: Int(seq_length) )
  var targets = Array<Int32>( repeating: 0, count: Int(seq_length) )

  var Wxh   = Tensor<Float>(randomNormal:[hidden_size, vocab_size]  )*0.01  // input to hidden
  var Whh   = Tensor<Float>(randomNormal:[hidden_size, hidden_size] )*0.01  // hidden to hidden
  var Why   = Tensor<Float>(randomNormal:[vocab_size,  hidden_size] )*0.01  // input to hidden
  var bh    = Tensor<Float>( zeros:[hidden_size, 1 ] )                      // hidden bias
  var by    = Tensor<Float>( zeros:[vocab_size,  1] )                       // output bias
  var hprev = Tensor<Float>( zeros:TensorShape( hidden_size, 1 ) )          // previous hidden state


  //gradients
  var dWxh   = Tensor<Float>(zeros:Wxh.shape)
  var dWhh   = Tensor<Float>(zeros:Whh.shape)
  var dWhy   = Tensor<Float>(zeros:Why.shape)
  var dbh    = Tensor<Float>(zeros:bh.shape)
  var dby    = Tensor<Float>(zeros:by.shape)
  var dhnext = Tensor<Float>(zeros:hprev.shape)


  var mWxh = Tensor<Float>(zeros:Wxh.shape)
  var mWhh = Tensor<Float>(zeros:Whh.shape)
  var mWhy = Tensor<Float>(zeros:Why.shape)
  var mbh  = Tensor<Float>(zeros:bh.shape ) // memory variables for Adagrad
  var mby  = Tensor<Float>(zeros:by.shape ) // memory variables for Adagrad

  let vocab_zeros = Tensor<Float>(zeros:[vocab_size,1] )
  let one         = Tensor<Float>(ones:[1])

  var loss  = Tensor<Float>(zeros:[1])

  func lossFun(inputs: Array<Int32>, targets: Array<Int32> ) {
    //calculate the loss, gradients on model parameters, and last hidden state
    
    var xs = [Int: Tensor<Float>]()
    var hs = [Int(-1): hprev]
    var ys = [Int: Tensor<Float>]()
    var ps = [Int: Tensor<Float>]()
    loss *= 0
    
    // forward pass for each charater index in inputs
    for t in 0..<inputs.count {
      let in_token  = inputs[t]
      let out_token = targets[t]
  
      // encode in 1-of-k representation
      xs[t] = vocab_zeros
      xs[t]![in_token] = one
      
      // new hidden state = tanh of contribution from: input layer + current hidden state + bias
      hs[t] = tanh(matmul(Wxh,xs[t]!) + matmul(Whh,hs[t-1]!) + bh)
      ys[t] = matmul(Why, hs[t]!) + by                // unnormalized log probabilities for next chars

      ps[t] = exp(ys[t]!) / exp(ys[t]!).sum()         // probabilities for next chars
    
      loss += -log(ps[t]![out_token][0])              // softmax (cross-entropy loss)
    }
    hprev = hs[inputs.count-1]!


    // backward pass: compute gradients going backwards
    dWxh   *= 0.0
    dWhh   *= 0.0
    dWhy   *= 0.0
    dbh    *= 0.0
    dby    *= 0.0
    dhnext *= 0.0
  
    for t in (0..<inputs.count).reversed() {
      var dy          = ps[t]!
      dy[targets[t]] -= 1   // backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    
      dWhy  += matmul(dy, hs[t]!.transposed() )
      dby   += dy
    
      let dh    = matmul(Why.transposed(), dy) + dhnext  // backprop into h
      let dhraw = (1 - hs[t]! * hs[t]!) * dh             // backprop through tanh nonlinearity
    
      dbh   += dhraw
      dWxh  += matmul(dhraw, xs[t]!.transposed())
      dWhh  += matmul(dhraw, hs[t-1]!.transposed())
      dhnext = matmul(Whh.transposed(), dhraw)
    }
  
    // clip to mitigate exploding gradients
    dWxh = Raw.clipByValue(t: dWxh, clipValueMin: Tensor<Float>(-5), clipValueMax: Tensor<Float>(5))
    dWhh = Raw.clipByValue(t: dWhh, clipValueMin: Tensor<Float>(-5), clipValueMax: Tensor<Float>(5))
    dWhy = Raw.clipByValue(t: dWhy, clipValueMin: Tensor<Float>(-5), clipValueMax: Tensor<Float>(5))
    dbh  = Raw.clipByValue(t: dbh,  clipValueMin: Tensor<Float>(-5), clipValueMax: Tensor<Float>(5))
    dby  = Raw.clipByValue(t: dby,  clipValueMin: Tensor<Float>(-5), clipValueMax: Tensor<Float>(5))
  }

  /*func sample(h:Tensor<Float>, seed_ix:Int32, n:Int) -> Array<Int32> {
    //sample a sequence of integers from the model
    //h is memory state, seed_ix is seed letter for first time step
    var ixes = Array<Int32>()
    var ix   = seed_ix
    for _ in 0..<n {
      var x    = vocab_zeros
      x[ix] = one
      ixes.append( ix )

      //forward
      let h  = tanh(matmul(Wxh, x) + matmul(Whh, h) + bh)
      let y  = matmul(Why, h) + by
      let p  = exp(y) / exp(y).sum()
    
      //pick the most likely character. ie index of (max(p). very primitive approach
      var mx = Tensor<Float>(0)
      for i in 0..<p.scalarCount {
        if mx < p[i] {
          mx = p[i]
          ix = Int32(i)
        }
      }
    }
    return ixes
  }*/


  var smooth_loss = Tensor<Float>(repeating: -log( Float(1) / Float( vocab_size ) ) * Float(seq_length), shape: TensorShape(1) )  // loss at iteration 0
  var n : Int  = 0
  var p : Int = 0

  while true {
  //while n<=0{
  
    //reset RNN memory at start of each epoch
    if p+Int(seq_length)+1 >= data.count || n == 0 {
      hprev *= 0.0 // reset RNN memory
      p      = 0   // go from start of data
    }

    // load inputs and target
    for i in 0..<Int(seq_length) {
      inputs[i]  = char_to_ix[ data[p+i] ]!
      targets[i] = char_to_ix[ data[p+i+1] ]!
    }
  
    /*// sample from the model now and then
    if n % 100 == 0 {
      var chars     = Array<Character>()
      let sample_ix = sample(h: hprev, seed_ix: char_to_ix["T"]!, n: 100)
      for s in sample_ix {
        chars.append( ix_to_char[s]! )
      }
      print( "----\n \(String( chars )) \n----")
    }*/
  
    // forward seq_length characters through the net and fetch gradient
    lossFun(inputs:inputs, targets:targets)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 100 == 0 {
      print("iter \(n), loss: \(smooth_loss)") // print progress
    }
  
    //update weights and bias
    mWxh +=  dWxh * dWxh
    Wxh  += -learning_rate * dWxh  / sqrt(mWxh + 1e-8) // adagrad update
    mWhh +=  dWhh * dWhh
    Whh  += -learning_rate * dWhh  / sqrt(mWhh + 1e-8) // adagrad update
    mWhy +=  dWhy * dWhy
    Why  += -learning_rate * dWhy  / sqrt(mWhy + 1e-8) // adagrad update
    mbh  +=  dbh * dbh
    bh   += -learning_rate * dbh   / sqrt(mbh + 1e-8) // adagrad update
    mby  +=  dby * dby
    by   += -learning_rate * dby   / sqrt(mby + 1e-8) // adagrad update

    p += Int(seq_length)  // move data pointer
    n += 1                // iteration counter
  }
}

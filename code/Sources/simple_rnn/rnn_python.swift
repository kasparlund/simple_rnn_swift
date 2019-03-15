import Python

func rnn_python(data:Array<Character>){
  print("python version: \(Python.versionInfo)" )
  let sys = Python.import("sys")
  sys.path.append("/anaconda3/lib/python3.6/site-packages") //xcode ignore conda environments so we set the path here
  let np = Python.import("numpy")
  let gc = Python.import("gc")

  print("Learning sherlock_holmes using a simple RNN with python");
  let chars      = Set<Character> ( data )
  let data_size  = data.count
  let vocab_size = chars.count
  print("data has \(data_size) characters with, \(vocab_size) unique characters")

  var char_to_ix = [Character: Int]()
  var ix_to_char = [Int: Character]()
  for (i,c) in chars.enumerated() {
    char_to_ix[c] = i
    ix_to_char[i] = c
    print(c) //print character in vocabulary
  }

  // hyperparameters
  let hidden_size = 100     // size of hidden layer of neurons
  let seq_length  = 25      // bptt: number of steps to unroll the RNN for
  let learning_rate : PythonObject = 1e-1

  var inputs  = Array<Int>( repeating: 0, count: seq_length)
  var targets = Array<Int>( repeating: 0, count: seq_length)

  var Wxh = np.random.randn(hidden_size, vocab_size)*0.01  // input to hidden
  var Whh = np.random.randn(hidden_size, hidden_size)*0.01 // hidden to hidden
  var Why = np.random.randn(vocab_size,  hidden_size)*0.01 // hidden to output
  var bh  = np.zeros([hidden_size, 1])                     // hidden bias
  var by  = np.zeros([vocab_size,  1])                     // output bias

  var hprev  = np.zeros( [hidden_size, 1] )                // previous hidden state


  //gradients
  var dWxh   = np.zeros_like(Wxh)
  var dWhh   = np.zeros_like(Whh)
  var dWhy   = np.zeros_like(Why)
  var dbh    = np.zeros_like(bh)
  var dby    = np.zeros_like(by)
  var dhnext = np.zeros( [hidden_size, 1] )
  var loss : Float = 0


  func lossFun(inputs: Array<Int>, targets: Array<Int> ) {
    
    //inputs,targets are both list of integers.
    //hprev is Hx1 array of initial hidden state
    //returns the loss, gradients on model parameters, and last hidden state
    var xs = [Int: PythonObject]()
    var hs = [Int(-1): hprev]
    var ys = [Int: PythonObject]()
    var ps = [Int: PythonObject]()
    loss *= 0

    // forward pass for each charater index in inputs
    for t in 0..<inputs.count {
      let v = np.zeros([vocab_size,1])      // encode in 1-of-k representation
      v[inputs[t]] = 1
      xs[t] = v
      
      hs[t] = np.tanh( np.dot(Wxh, xs[t]!) + np.dot(Whh, hs[t-1]!) + bh )  // hidden state
      
      ys[t] = np.dot(Why, hs[t]!) + by                // unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]!) / np.sum(np.exp(ys[t]!)) // probabilities for next chars
      
      let v_ps = ps[t]![targets[t],0]
      loss += -Float(np.log(v_ps))!                  // softmax (cross-entropy loss)
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
      let dy  = np.copy( ps[t]! )
      dy[targets[t]] -= 1   // backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
      
      dWhy  += np.dot(dy, hs[t]!.T )
      dby   += dy
      let dh    = np.dot(Why.T, dy) + dhnext  // backprop into h
      let dhraw = (1 - hs[t]! * hs[t]!) * dh  // backprop through tanh nonlinearity
      dbh   += dhraw
      dWxh  += np.dot(dhraw, xs[t]!.T)
      dWhh  += np.dot(dhraw, hs[t-1]!.T)
      dhnext = np.dot(Whh.T, dhraw)
    }

    // clip to mitigate exploding gradients
    np.clip(dWxh, -5, 5, out:dWxh )
    np.clip(dWxh, -5, 5, out:dWxh )
    np.clip(dWhh, -5, 5, out:dWhh )
    np.clip(dWhy, -5, 5, out:dWhy )
    np.clip(dbh,  -5, 5, out:dbh  )
    np.clip(dby,  -5, 5, out:dby  )
    
  }
  
  /*func sample(h:PythonObject, seed_ix:Int, n:Int) -> Array<Int> {
    //sample a sequence of integers from the model
    //h is memory state, seed_ix is seed letter for first time step
    var ixes = Array<Int>()
    var ix   = seed_ix

    for _ in 0..<n {
      let x = np.zeros([vocab_size, 1])
      x[ix] = 1
      
      let h  = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
      let y  = np.dot(Why, h) + by
      let p  = np.exp(y) / np.sum(np.exp(y))
      
      ix = Int(np.random.choice( vocab_size, p:p.ravel()))!
      ixes.append( ix )
    }
    return ixes
  }*/


  let mWxh = np.zeros_like(Wxh)
  let mWhh = np.zeros_like(Whh)
  let mWhy = np.zeros_like(Why)
  let mbh  = np.zeros_like(bh) // memory variables for Adagrad
  let mby  = np.zeros_like(by) // memory variables for Adagrad

  var smooth_loss = -Float(np.log( 1.0/Double(vocab_size) ) )! * Float(seq_length) // loss at iteration 0


  var n : Int = 0
  var p : Int = 0
  while true {
  //while p<100000 {
    
    //reset RNN memory at start of each epoch
    if p+seq_length+1 >= data.count || n == 0 {
      hprev *= 0.0 // reset RNN memory
      p      = 0   // go from start of data
    }

    // load inputs and target
    for i in 0..<seq_length {
      inputs[i]  = char_to_ix[ data[p+i] ]!
      targets[i] = char_to_ix[ data[p+i+1] ]!
    }
    
    /*
    // sample from the model now and then.
    if n % 100 == 0 {
      let seed_ix   = inputs[0]
      let sample_ix = sample(h: hprev, seed_ix: seed_ix, n: 200)
      var chars     = Array<Character>()
      for s in sample_ix {
        chars.append( ix_to_char[sample_ix[s]]! )
      }
      let txt = String( chars )
      print( "----\n \(txt) \n----")
    }
    */
    // forward seq_length characters through the net and fetch gradient
    lossFun(inputs:inputs, targets:targets)
    
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0 {
      print("iter \(n), loss: \(smooth_loss)") // print progress
      gc.collect()
    }
    
    //update weights and bias
    var params  = [Wxh, Whh, Why, bh, by]
    var dparams = [dWxh, dWhh, dWhy, dbh, dby]
    var mems    = [mWxh, mWhh, mWhy, mbh, mby]
    for i in 0..<params.count {
      mems[i]   += dparams[i] * dparams[i]
      params[i] += -learning_rate * dparams[i]  / np.sqrt(mems[i] + 1e-8) // adagrad update
    }

    p += seq_length // move data pointer
    n += 1          // iteration counter
  }
}

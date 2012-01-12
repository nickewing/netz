(in-ns 'nn.core)

(defn- bind-bias
  "Prepend the bias unit to a layer activation vector."
  [v]
  (bind-rows [1] v))

(defn- sigmoid
  "Sigmoid function: 1/(1+exp(-z))."
  [z]
  (div 1 (plus 1 (exp (minus z)))))

(defn- forward-propagate-layer
  "Calculate activations for layer l+1 given weight matrix between layer l
  and l+1 and layer l activations."
  [weight-matrix activations]
  (sigmoid (matrix-mult weight-matrix activations)))

(defn- forward-propagate-all-activations
  "Propagate activation values through the network and return all activation
  values for all nodes."
  [weights input-activations]
  (let [input-activations (bind-bias input-activations)]
    (loop [all-weights weights
           activations input-activations
           all-activations [input-activations]]
      (let [weights (first all-weights)
            all-weights (rest all-weights)
            last-iter (empty? all-weights)
            activations (forward-propagate-layer weights activations)
            activations (if last-iter
                          activations
                          (bind-bias activations))
            all-activations (conj all-activations activations)]
        (if last-iter
          all-activations
          (recur all-weights activations all-activations))))))

(defn- forward-propagate
  "Propagate activation values through the network and return output layer
  activation values."
  [weights input-activations]
  (reduce #(forward-propagate-layer %2 (bind-bias %1)) input-activations weights))

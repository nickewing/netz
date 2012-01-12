(in-ns 'nn.core)

(def init-epsilon 0.5)

(defn- random-list
  "Create a list of random doubles between -init-epsilon and +init-epsilon."
  [len]
  (for [x (range 0 len)]
    (- (rand (* 2 init-epsilon)) init-epsilon)))

(defn- random-weight-matrices
  "Generate random initial weight matrices for given layer-sizes."
  [layer-sizes]
  (for [i (range 0 (dec (length layer-sizes)))]
    (let [cols (inc (get layer-sizes i))
          rows (get layer-sizes (inc i))]
      (matrix (random-list (* rows cols)) cols))))

(defn- back-propagate-layer-deltas
  "Back propagate last-deltas (from layer l-1) and return layer l deltas."
  [last-deltas weight-matrix layer-activations]
  (mult (matrix-mult (trans weight-matrix) last-deltas)
        (mult layer-activations (minus 1 layer-activations))))

(defn- calc-hidden-deltas
  "Calculate hidden deltas for back propagation.  Returns all deltas including
  output-deltas."
  [weights activations output-deltas]
  (let [hidden-weights (reverse (rest weights))
        hidden-activations (rest (reverse (rest activations)))]
    (loop [last-deltas output-deltas
           all-weights hidden-weights
           all-activations hidden-activations
           all-deltas (list output-deltas)]
      (if (empty? all-weights)
        all-deltas
        (let [weights (first all-weights)
              all-weights (rest all-weights)
              activations (first all-activations)
              all-activations (rest all-activations)
              last-deltas (back-propagate-layer-deltas last-deltas weights activations)
              all-deltas (cons (rest last-deltas) all-deltas)]
          (recur last-deltas all-weights all-activations all-deltas))))))

(defn- add-delta-sum
  "Add new delta sum to accumulator."
  [delta-sums deltas activations]
  (map #(plus %1 (mmult %2 (trans %3)))
       delta-sums deltas activations))

(defn- new-weight-accumulator
  "Create accumulator matrix list of the same structure as the given weight list
  with all zero values."
  [weights]
  (map (fn [weight]
         (let [[rows cols] (dim weight)]
           (matrix 0 rows cols)))
       weights))

(defn- back-propagate-example
  "Run back propagation on one sample and return delta sums."
  [delta-sums weights input expected-output]
  (let [activations (forward-propagate-all-activations weights (matrix input))
        output (last activations)
        output-deltas (minus output expected-output)
        all-deltas (calc-hidden-deltas weights activations output-deltas)
        delta-sums (add-delta-sum delta-sums all-deltas activations)]
    delta-sums))

(defn- regularize-gradients
  "gradient = gradient + lambda * weights for all columns except the first."
  [network weights gradients]
  (map (fn [gradients weights]
         (let [[rows cols] (dim weights)]
           (plus gradients
                 (bind-columns
                   (matrix 0 rows 1)
                   (mult (option network :regularization-constant)
                         (sel weights :except-cols 0))))))
       gradients
       weights))

(defn- back-propagate-all-examples
  "Run back propagation on all examples and return gradients."
  [network weights examples]
  (let [delta-sums (reduce
                     #(back-propagate-example %1 weights (first %2) (second %2))
                     (new-weight-accumulator weights)
                     examples)]
    (regularize-gradients
      network
      weights
      (map #(div % (length examples)) delta-sums))))

(defn- calc-weight-changes
  "Calculate weight changes:
  changes = learning rate * gradients + last change * learning momentum."
  [network gradients last-changes]
  (map #(plus (mult (option network :learning-rate) %1)
              (mult (option network :learning-momentum) %2))
       gradients
       last-changes))

(defn- apply-weight-changes
  "Applies changes to weights:
  ∀(weights, changes) weights := weights - changes."
  [weights changes]
  (map #(minus %1 %2) weights changes))


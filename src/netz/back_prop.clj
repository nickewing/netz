(in-ns 'netz.core)

(import '(java.util.concurrent Executors))

(declare ^:dynamic *thread-pool*)

(def init-epsilon 0.5)

(defn- new-thread-pool []
  (Executors/newFixedThreadPool
    (.. Runtime getRuntime availableProcessors)))

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

(defn- calc-delta-sum
  "Add new delta sum to accumulator."
  [deltas activations]
  (map #(mmult %1 (trans %2)) deltas activations))

(defn- new-weight-accumulator
  "Create accumulator matrix list of the same structure as the given weight list
  with all zero values."
  [weights]
  (map #(let [[rows cols] (dim %)]
          (matrix 0 rows cols))
       weights))

(defn- add-to-accumulator
  [accumulator change]
  (map #(plus %1 %2) accumulator change))

(defn- calc-example-error
  "Calculate deltas and squared error for given example."
  [delta-sums weights [input expected-output]]
  (let [activations (forward-propagate-all-activations weights (matrix input))
        output (last activations)
        output-deltas (minus output expected-output)
        all-deltas (calc-hidden-deltas weights activations output-deltas)
        delta-sum (calc-delta-sum all-deltas activations)]
    (list delta-sum
          (sum (pow output-deltas 2)))))

(defn- regularize-gradients
  "gradient = gradient + lambda * weights for all columns except the first."
  [network gradients]
  (let [regularization-constant (network-option network-option :regularization-constant)]
    (if regularization-constant
      (map (fn [gradients weights]
             (let [[rows cols] (dim weights)]
               (plus gradients
                     (bind-columns
                       (matrix 0 rows 1)
                       (mult regularization-constant
                             (sel weights :except-cols 0))))))
           gradients
           (:weights network))
      gradients)))

(defn- calc-batch-error-sequential
  [network examples]
  (let [weights (:weights network)]
    (loop [delta-sums (new-weight-accumulator weights)
           total-error 0
           examples examples]
      (let [example (first examples)
            examples (rest examples)
            [delta-sum squared-error] (calc-example-error delta-sums weights example)
            delta-sums (add-to-accumulator delta-sums delta-sum)
            total-error (+ total-error squared-error)]
        (if (empty? examples)
          (list delta-sums total-error)
          (recur delta-sums total-error examples))))))

(defn- calc-group-error-parallel
  [network examples total-error delta-sums]
  (let [weights (:weights network)]
    (loop [examples examples]
      (let [example (first examples)
            examples (rest examples)
            [delta-sum squared-error] (calc-example-error delta-sums weights example)]
        (swap! total-error (partial + squared-error))
        (swap! delta-sums (partial add-to-accumulator delta-sum))
        (if-not (empty? examples)
          (recur examples))))))

(defn- calc-batch-error-parallel
  [network examples]
  (let [weights (:weights network)
        total-error (atom 0)
        delta-sums (atom (new-weight-accumulator weights))
        groups (partition-all (.getMaximumPoolSize *thread-pool*) examples)
        tasks (map
                (fn [group]
                  #(calc-group-error-parallel network group total-error delta-sums))
                groups)]
    (doseq [future (.invokeAll *thread-pool* tasks)]
      (.get future))
    (list @delta-sums @total-error)))

(defn- calc-batch-error
  "Calculate gradients and MSE for example set."
  [network examples]
  (let [calc-error-fn (if (network-option network :calc-batch-error-in-parallel)
                        calc-batch-error-parallel
                        calc-batch-error-sequential)
        num-examples (length examples)
        [delta-sums total-error] (calc-error-fn network examples)]
    (list
      ; gradients
      (regularize-gradients network (map #(div % num-examples) delta-sums))
      ; mean squared error
      (/ total-error num-examples))))

(defn- calc-weight-changes
  "Calculate weight changes:
  changes = learning rate * gradients + last change * learning momentum."
  [network gradients last-changes]
  (map #(plus (mult (network-option network :learning-rate) %1)
              (mult (network-option network :learning-momentum) %2))
       gradients
       last-changes))

(defn- apply-weight-changes
  "Applies changes to weights:
  ∀(weights, changes) weights := weights - changes."
  [weights changes]
  (map #(minus %1 %2) weights changes))


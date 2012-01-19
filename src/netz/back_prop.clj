(in-ns 'netz.core)

(import '(java.util.concurrent Executors))

(defn- new-thread-pool []
  (Executors/newFixedThreadPool
    (.. Runtime getRuntime availableProcessors)))

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
              activations (first all-activations)
              all-weights (rest all-weights)
              all-activations (rest all-activations)
              last-deltas (back-propagate-layer-deltas last-deltas weights activations)
              all-deltas (cons (rest last-deltas) all-deltas)]
          (recur last-deltas all-weights all-activations all-deltas))))))

(defn- calc-gradient-sum
  "Calculate delta sum"
  [deltas activations]
  (map #(mmult %1 (trans %2)) deltas activations))

(defn- new-synapse-list
  "Create accumulator matrix list of the same structure as the given weight list
  with all elements set to value."
  [weights value]
  (map #(let [[rows cols] (dim %)]
          (matrix value rows cols))
       weights))

(defn- add-to-synapse-list
  [synapse-list change]
  (map #(plus %1 %2) synapse-list change))

(defn- calc-example-error
  "Calculate deltas and squared error for given example."
  [gradient-sums weights [input expected-output]]
  (let [activations (forward-propagate-all-activations weights (matrix input))
        output (last activations)
        output-deltas (minus output expected-output)
        all-deltas (calc-hidden-deltas weights activations output-deltas)
        gradient-sum (calc-gradient-sum all-deltas activations)]
    (list gradient-sum
          (sum (pow output-deltas 2)))))

(defn- regularize-gradients
  "gradient = gradient + lambda * weights for all columns except the first."
  [network gradients regularization-constant]
  (if (not= regularization-constant 0)
    (map (fn [gradients weights]
           (let [[rows cols] (dim weights)]
             (plus gradients
                   (bind-columns
                     (matrix 0 rows 1)
                     (mult regularization-constant
                           (sel weights :except-cols 0))))))
         gradients
         (:weights network))
    gradients))

(defn- calc-batch-error-sequential
  [network examples]
  (let [weights (:weights network)]
    (loop [gradient-sums (new-synapse-list weights 0)
           total-error 0
           examples examples]
      (let [example (first examples)
            examples (rest examples)
            [gradient-sum squared-error] (calc-example-error gradient-sums weights example)
            gradient-sums (add-to-synapse-list gradient-sums gradient-sum)
            total-error (+ total-error squared-error)]
        (if (empty? examples)
          (list gradient-sums total-error)
          (recur gradient-sums total-error examples))))))

(defn- calc-group-error-parallel
  [network examples total-error gradient-sums]
  (let [weights (:weights network)]
    (doseq [example examples]
      (let [[gradient-sum squared-error] (calc-example-error gradient-sums weights example)]
        (swap! total-error (partial + squared-error))
        (swap! gradient-sums (partial add-to-synapse-list gradient-sum))))))

(defn- calc-batch-error-parallel
  [thread-pool network examples]
  (let [weights (:weights network)
        total-error (atom 0)
        gradient-sums (atom (new-synapse-list weights 0))
        groups (partition-all (.getMaximumPoolSize thread-pool) examples)
        tasks (map
                (fn [group]
                  #(calc-group-error-parallel network group total-error gradient-sums))
                groups)]
    (doseq [future (.invokeAll thread-pool tasks)]
      (.get future))
    (list @gradient-sums @total-error)))

(defn- calc-batch-error
  "Calculate gradients and MSE for example set."
  [network examples regularization-constant thread-pool]
  (let [calc-error-fn (if thread-pool
                        (partial calc-batch-error-parallel thread-pool)
                        calc-batch-error-sequential)
        num-examples (length examples)
        [gradient-sums total-error] (calc-error-fn network examples)]
    (list
      ; gradients
      (regularize-gradients network
                            (map #(div % num-examples) gradient-sums)
                            regularization-constant)
      ; mean squared error
      (/ total-error num-examples))))

(defn- bprop-calc-weight-changes
  "Calculate weight changes:
  changes = learning rate * gradients + last change * learning momentum."
  [network gradients last-changes options]
  (let [learning-rate (:learning-rate options)
        learning-momentum (:learning-momentum options)]
    (map #(plus (mult learning-rate %1)
                (mult learning-momentum %2))
         gradients
         last-changes)))

(defn- apply-weight-changes
  "Applies changes to weights:
  ∀(weights, changes) weights := weights - changes."
  [weights changes]
  (map #(plus %1 %2) weights changes))

(defn- rprop-calc-weight-change
  [last-gradient gradient last-update inc-factor dec-factor update-min update-max]
  (let [[rows cols] (dim gradient)
        new-last-gradient (copy gradient)
        change (matrix 0 rows cols)
        update (copy last-update)]
    (dotimes [i rows]
      (dotimes [j cols]
        (let [last-gradient-elem (.getQuick last-gradient i j)
              gradient-elem (.getQuick gradient i j)
              last-update-elem (.getQuick last-update i j)
              dir (* last-gradient-elem gradient-elem)]
          (cond
            (> dir 0.0) (let [update-elem (min
                                            (* last-update-elem inc-factor)
                                            update-max)]
                          (.setQuick update i j update-elem)
                          (.setQuick change i j (* (- (sign gradient-elem)) update-elem))
                          )
            (< dir 0.0) (do
                          (.setQuick update i j (max (* last-update-elem dec-factor)
                                                     update-min))
                          (.setQuick change i j 0)
                          (.setQuick new-last-gradient i j 0))
            (= dir 0.0) (do
                          (.setQuick change i j (* (- (sign gradient-elem)) last-update-elem)))))))
    (list new-last-gradient update change)))

(defn- rprop-calc-weight-changes
  [last-gradients gradients last-updates options]
  (let [inc-factor (:increase-factor options)
        dec-factor (:decrease-factor options)
        update-min (:update-min options)
        update-max (:update-max options)]
    (map-to-vectors
      #(rprop-calc-weight-change %1 %2 %3 inc-factor dec-factor update-min update-max)
      last-gradients
      gradients
      last-updates)))


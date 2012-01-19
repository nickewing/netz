(in-ns 'netz.core)

(defn- gradient-descent-complete?
  [network epoch mse]
  (or (>= epoch (get-option network :max-epochs))
      (< mse (get-option network :desired-error))))

(defn- call-callback-for-epoch
  [network epoch mse complete]
  (let [callback (get-option network :callback)
        callback-resolution (get-option network :callback-resolution)]
    (if (and callback
             (or complete
                 (= (mod epoch callback-resolution) 0)))
      (callback network epoch mse complete)
      true)))

(defn- gradient-descent
  "Preform gradient descent to adjust network weights"
  [step-fn init-state network examples]
  (let [calc-batch-error-in-parallel (get-option network :calc-batch-error-in-parallel)
        regularize-gradients (get-option network :regularization-constant)
        thread-pool (if calc-batch-error-in-parallel (new-thread-pool))]
    (loop [network network
           state init-state
           epoch 0]
      (let [epoch (inc epoch)
            [gradients mse] (calc-batch-error network
                                              examples
                                              regularize-gradients
                                              thread-pool)]
        (if (or (gradient-descent-complete? network epoch mse)
                (not (call-callback-for-epoch network epoch mse false)))
          (do
            (call-callback-for-epoch network epoch mse true)
            network)
          (let [[changes state] (step-fn network gradients state)
                new-weights (apply-weight-changes (:weights network) changes)
                network (assoc network :weights new-weights)]
            (recur network state epoch)))))))

(defn gradient-descent-bprop
  [network examples]
  (let [options (get-option network :bprop)
        last-changes (new-synapse-list (:weights network) 0)]
    (gradient-descent
      (fn [network gradients last-changes]
        (let [changes (bprop-calc-weight-changes network gradients last-changes options)]
          [(map minus changes) changes]))
      last-changes
      network
      examples)))

(defn gradient-descent-rprop
  [network examples]
  (let [options (get-option network :rprop)
        last-gradients (new-synapse-list (:weights network) 0)
        last-updates (new-synapse-list (:weights network) (:init-update options))]
    (gradient-descent
      (fn [network gradients [last-gradients last-updates]]
        (let [[last-gradients updates changes]
              (rprop-calc-weight-changes last-gradients gradients last-updates options)]
          [changes [last-gradients updates]]))
      (list last-gradients last-updates)
      network
      examples)))

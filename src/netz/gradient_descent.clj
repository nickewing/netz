(in-ns 'netz.core)

(defn- gradient-descent-complete?
  [network epoch mse]
  (or (>= epoch (option network :max-epochs))
      (< mse (option network :desired-error))))

(defn- call-callback-for-epoch
  [callback epochs-per-callback epoch mse]
  (if (and callback
           (= (mod epoch epochs-per-callback) 0))
    (callback epoch mse false)
    true))

(defn- gradient-descent
  "Preform gradient descent to adjust network weights"
  [network examples]
  (let [callback (option network :callback)
        epochs-per-callback 100]
    (loop [weights (:weights network)
           last-changes (new-weight-accumulator (:weights network))
           epoch 0]
      (let [epoch (inc epoch)
            [gradients mse] (calc-all-errors network weights examples)]
        (if (or (gradient-descent-complete? network epoch mse)
                (not (call-callback-for-epoch callback epochs-per-callback epoch mse)))
          (do
            (if callback
              (callback epoch mse true))
            (assoc network :weights weights))
          (let [changes (calc-weight-changes network gradients last-changes)
                new-weights (apply-weight-changes weights changes)]
            (recur new-weights changes epoch)))))))

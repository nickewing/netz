(in-ns 'netz.core)

(defn- gradient-descent-complete?
  [network epoch mse]
  (or (>= epoch (network-option network :max-epochs))
      (< mse (network-option network :desired-error))))

(defn- call-callback-for-epoch
  [network epoch mse complete]
  (let [callback  (network-option network :callback)
        callback-resolution (network-option network :callback-resolution)]
    (if (and callback
             (or complete
                 (= (mod epoch callback-resolution) 0)))
      (callback network epoch mse complete)
      true)))

(defn- gradient-descent-backpropagation
  "Preform gradient descent backpropagation to adjust network weights"
  [network examples]
  (loop [network network
         last-changes (new-weight-accumulator (:weights network))
         epoch 0]
    (let [epoch (inc epoch)
          [gradients mse] (calc-all-errors network examples)]
      (if (or (gradient-descent-complete? network epoch mse)
              (not (call-callback-for-epoch network epoch mse false)))
        (do
          (call-callback-for-epoch network epoch mse true)
          network)
        (let [changes (calc-weight-changes network gradients last-changes)
              new-weights (apply-weight-changes (:weights network) changes)
              network (assoc network :weights new-weights)]
          (recur network changes epoch))))))

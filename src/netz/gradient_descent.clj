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

(defn- gradient-descent
  "Preform gradient descent backpropagation to adjust network weights"
  [init-fn step-fn network examples]
  (with-new-thread-pool network
    (loop [network network
           state (init-fn network)
           epoch 0]
      (let [epoch (inc epoch)
            [gradients mse] (calc-batch-error network examples)]
        (if (or (gradient-descent-complete? network epoch mse)
                (not (call-callback-for-epoch network epoch mse false)))
          (do
            (call-callback-for-epoch network epoch mse true)
            network)
          (let [[changes state] (step-fn network gradients state)
                ; _ (if (= (mod epoch (network-option network :callback-resolution)) 0)
                    ; (println changes))
                new-weights (apply-weight-changes (:weights network) changes)
                network (assoc network :weights new-weights)]
            (recur network state epoch)))))))

(defn- bprop-init-state [network]
  (new-synapse-list (:weights network) 0))

(defn- bprop-step [network gradients last-changes]
  (let [changes (calc-weight-changes network gradients last-changes)]
    [(map minus changes) changes]))

(def gradient-descent-bprop
  (partial gradient-descent bprop-init-state bprop-step))

(defn- rprop-init-state [network]
  (vector 
    (new-synapse-list (:weights network) 0)
    (new-synapse-list (:weights network)
                      (network-option network :rprop-init-update))))

(defn- rprop-step [network gradients [last-gradients last-updates]]
  ; (println "weights" (map to-vect (:weights network)))
  ; (println "gradients" (map to-vect gradients))
  ; (println "last-gradients" (map to-vect last-gradients))
  ; (println "product" (map to-vect (map mult gradients last-gradients)))
  ; (println "last-updates" (map to-vect last-updates))
  (let [[last-gradients updates changes]
            (rprop-calc-weight-changes network last-gradients gradients last-updates)]
    ; (println "> last-gradients" (map to-vect last-gradients))
    ; (println "> updates" (map to-vect updates))
    ; (println "> changes" (map to-vect changes))
    ; (println "===")
    [changes [last-gradients updates]]))

(def gradient-descent-rprop
  (partial gradient-descent rprop-init-state rprop-step))



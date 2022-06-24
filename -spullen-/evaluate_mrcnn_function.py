def evaluate_mrcnn(model, rgb):
    print("\nRecognition phase...")
    start = time.time()

    results = model.detect([rgb], verbose=0)
    r = results[0]
    box, mask, classID, score = r['rois'], r['masks'], r['class_ids'], r['scores']                      

    end = time.time()
    print('MRCNN execution time: ', end - start)

    return box, mask, classID, score
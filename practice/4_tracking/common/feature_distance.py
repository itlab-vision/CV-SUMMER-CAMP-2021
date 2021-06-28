from scipy.spatial.distance import cosine as cosine_distance

def calc_features_similarity(feature1, feature2):
    d = cosine_distance(feature1, feature2)
    sim = 1 - d
    if sim < 0:
        sim = 0
    assert 0 <= sim <= 1
    return sim


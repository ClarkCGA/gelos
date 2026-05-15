__all__ = ["generate_cloud_embeddings"]


def __getattr__(name):
    if name == "generate_cloud_embeddings":
        from gelos.cloud_embeddings.runner import generate_cloud_embeddings

        return generate_cloud_embeddings
    raise AttributeError(f"module 'gelos.cloud_embeddings' has no attribute {name!r}")

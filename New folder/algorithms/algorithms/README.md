# Algorithms

All four team algorithms live at the same level in this directory and are
called by the shared root-level `app.py`.

```text
algorithms/
|-- rhlt/          # Member 1 - current working implementation
|-- algorithm_2/   # Member 2 - reserved
|-- algorithm_3/   # Member 3 - reserved
`-- algorithm_4/   # Member 4 - reserved
```

## Shared integration contract

Each algorithm should expose its public processing function from its package
`__init__.py`. The function should accept an RGB or grayscale NumPy image and
return at least the following common values:

```python
{
    "algorithm_name": "Display name",
    "enhanced_image": image_array,
    "metrics": {
        "processing_time_ms": 0.0,
    },
}
```

An algorithm may return extra intermediate images and algorithm-specific
metrics. When the other implementations are ready, `app.py` can call each
available package with the same uploaded image and build a side-by-side table.

Do not place a separate Streamlit app inside an algorithm folder. There is only
one shared website: the root-level `app.py`.


# Vantaanjoki catchment area

Computes the upstream catchment area for any point in the Vantaanjoki river
basin (ETRS-TM35FIN / EPSG:3067 coordinates).

    python catchment_area.py --xy 369002.59,6704730.30

## Rasters (not included)

Too large for GitHub; place them in `04_accumulation/`:

- `flow_accumulation_vantaanjoki.tif`: 2 m D8 flow accumulation; each cell
  holds the upstream area draining through it, in m². Required; all areas
  come from it.
- `water_features_vantaanjoki.tif`: lake, peatland and wide-river-channel ids
  on the same grid. Optional; lets the tool name lakes and snap points to
  river channels.

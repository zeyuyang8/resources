---
title: Data Format
layout: default
parent: Craving
grand_parent: Datasets
---

## Data Format

### Fitbit

```bash
fitbit
├── cr001
│   ├── heart
│   ├── sleep
│   └── steps
├── cr002
│   ├── heart
│   ├── sleep
│   └── steps
├── ...
└── cr035
    ├── heart
    ├── sleep
    └── steps
```

We can extract the daily data of heart rate, sleep, and steps into CSV files of the following format:

| TIMESTAMP           | VALUE |
|:--------------------|:------|
| 2021-07-03 20:07:00 | 72    |
| 2021-07-03 20:08:00 | 75    |
| ...                 | ...   |

## GPU : debit median (Gpoints/s) et bande passante effective (16 octets/point)

| version | 256^3 | 512^3 | 768^3 |
|---|---|---|---|
| naive | 97.8 (1565 Go/s) | 100.0 (1600 Go/s) | 94.8 (1518 Go/s) |
| shared | 57.0 (913 Go/s) | 124.7 (1995 Go/s) | 109.2 (1748 Go/s) |
| sharedp | 56.9 (911 Go/s) | 122.3 (1956 Go/s) | 108.5 (1736 Go/s) |
| shared-32x4 | 62.1 (994 Go/s) | 125.3 (2004 Go/s) | 97.8 (1565 Go/s) |
| shared-32x16 | 62.5 (1000 Go/s) | 124.9 (1998 Go/s) | 112.1 (1793 Go/s) |
| shared-64x4 | 61.4 (982 Go/s) | 138.8 (2221 Go/s) | 125.3 (2005 Go/s) |
| shared-64x8 | 66.4 (1062 Go/s) | 142.3 (2276 Go/s) | 130.8 (2093 Go/s) |
| shared-zs | 116.2 (1860 Go/s) | 125.0 (2001 Go/s) | 109.3 (1748 Go/s) |
| shared-64x4-zs | 130.9 (2095 Go/s) | 141.1 (2257 Go/s) | 125.3 (2005 Go/s) |

## CPU Grace (512^3), OpenMP

| version, threads | Gpoints/s | Go/s eff. |
|---|---|---|
| blocked1 | 0.74 | 12 |
| blocked8 | 5.60 | 90 |
| blocked18 | 11.35 | 182 |
| blocked36 | 14.42 | 231 |
| blocked72 | 17.54 | 281 |
| flat1 | 0.78 | 12 |
| flat8 | 6.01 | 96 |
| flat18 | 8.04 | 129 |
| flat36 | 5.22 | 83 |
| flat72 | 8.28 | 132 |

Meilleur GPU / CPU 72 coeurs a 512^3 : 8x

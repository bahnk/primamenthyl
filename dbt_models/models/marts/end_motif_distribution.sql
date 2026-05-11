{{ config(materialized='table', tags=['merged']) }}

select
    sample,
    side,
    motif,
    count as motif_count
from {{ source('quant', 'quant__motif_counts') }}

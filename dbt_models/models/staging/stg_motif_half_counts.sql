{{ config(tags=['individual']) }}

select
    motif,
    side,
    round(count / 2.0) as count
from {{ source('quant', 'quant__motifs') }}

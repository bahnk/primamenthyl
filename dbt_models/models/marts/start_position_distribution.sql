{{ config(materialized='table', tags=['merged']) }}

select
    sample,
    start_position,
    count(*) as record_count
from {{ source('quant', 'quant__records') }}
group by sample, start_position

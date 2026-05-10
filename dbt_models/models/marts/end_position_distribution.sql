{{ config(materialized='table', tags=['merged']) }}

select
    sample,
    end_position,
    count(*) as record_count
from {{ source('quant', 'quant__records') }}
group by sample, end_position

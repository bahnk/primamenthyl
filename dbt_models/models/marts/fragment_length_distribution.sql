{{ config(materialized='table', tags=['merged']) }}

select
    sample,
    abs(template_length) as fragment_length,
    count(*) as record_count
from {{ source('quant', 'quant__records') }}
group by sample, abs(template_length)

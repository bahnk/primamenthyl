{{ config(tags=['individual']) }}

select distinct
    query_id,
    start_position,
    end_position
from {{ source('quant', 'quant__records') }}

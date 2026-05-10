{{ config(tags=['individual']) }}

select
    fragment_id,
    count(align_id) as align_id_count,
    count(align_id) = 2 as paired
from {{ ref('stg_record_fragments') }}
group by fragment_id

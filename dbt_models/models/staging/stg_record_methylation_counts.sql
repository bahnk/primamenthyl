{{ config(tags=['individual']) }}

select
    records.align_id as align_id,
    count(methylation.methylation_offset) as methylation_count
from {{ source('quant', 'quant__records') }} as records
left join {{ ref('stg_methylation_events') }} as methylation
    on records.align_id = methylation.align_id
group by records.align_id

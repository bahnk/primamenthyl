{{ config(tags=['individual']) }}

select
    methylation.align_id,
    records.reference,
    methylation.context,
    methylation.methylated,
    methylation.methylation_offset,
    records.position + methylation.methylation_offset as position
from {{ ref('stg_methylation_events') }} as methylation
inner join {{ source('quant', 'quant__records') }} as records
    on methylation.align_id = records.align_id

{{ config(tags=['individual']) }}

select
    methylation.align_id,
    records.reference,
    methylation.position as methylation_offset,
    records.position + methylation.position as position
from {{ source('quant', 'quant__methylation') }} as methylation
inner join {{ source('quant', 'quant__records') }} as records
    on methylation.align_id = records.align_id

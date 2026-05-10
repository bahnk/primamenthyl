select
    records.align_id as align_id,
    count(methylation.position) as methylation_count
from {{ source('quant', 'quant__records') }} as records
left join {{ source('quant', 'quant__methylation') }} as methylation
    on records.align_id = methylation.align_id
group by records.align_id

select
    records.read_id,
    count(methylation.position) as methylation_count
from {{ source('quant', 'quant__records') }} as records
left join {{ source('quant', 'quant__methylation') }} as methylation
    on records.read_id = methylation.read_id
group by records.read_id

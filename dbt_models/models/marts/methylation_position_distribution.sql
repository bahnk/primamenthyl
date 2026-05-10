{{ config(materialized='table', tags=['merged']) }}

select
    records.sample,
    records.reference,
    floor((records.position + methylation.position) / 100000) * 100000 as position_bin,
    count(*) as methylation_count
from {{ source('quant', 'quant__methylation') }} as methylation
inner join {{ source('quant', 'quant__records') }} as records
    on methylation.sample = records.sample
    and methylation.align_id = records.align_id
group by records.sample, records.reference, position_bin

{{ config(tags=['individual']) }}

with methylation_sites as (
    select
        reference,
        position,
        count(*)::double as meth
    from {{ ref('stg_methylation_positions') }}
    where context = 'cpg'
    group by reference, position
),

fragment_references as (
    select distinct
        query_id,
        reference
    from {{ source('quant', 'quant__records') }}
),

fragments as (
    select
        fragment_references.reference,
        query_fragments.query_id,
        query_fragments.start_position,
        query_fragments.end_position
    from {{ ref('stg_query_fragments') }} as query_fragments
    inner join fragment_references
        on query_fragments.query_id = fragment_references.query_id
)

select
    methylation_sites.reference as "CHR",
    methylation_sites.position as "START",
    methylation_sites.position + 1 as "END",
    methylation_sites.meth as "METH",
    count(fragments.query_id)::double as "DEPTH"
from methylation_sites
left join fragments
    on methylation_sites.reference = fragments.reference
    and methylation_sites.position between fragments.start_position and fragments.end_position
group by
    methylation_sites.reference,
    methylation_sites.position,
    methylation_sites.meth
